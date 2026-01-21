import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue, MatchAny
import uvicorn

# --- CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-xxxxxxxxxxxx")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333") 
COLLECTION_NAME = "shopify_products"

# --- INITIALIZATION ---
app = FastAPI(title="Advanced Product Recommender")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL)

# --- DATA MODELS ---
class FilterParams(BaseModel):
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    vendor: Optional[str] = None
    allowed_tags: Optional[List[str]] = None  # e.g. ["Blue", "Waterproof"]

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    filters: Optional[FilterParams] = None

class RecommendationRequest(BaseModel):
    positive_product_ids: List[int]
    negative_product_ids: List[int] = []
    limit: int = 5
    filters: Optional[FilterParams] = None

class SimilarRequest(BaseModel):
    product_id: int
    limit: int = 5
    filters: Optional[FilterParams] = None

# --- HELPER FUNCTIONS ---
def get_embedding(text: str):
    """Generates vector embedding using the same model as ingestion."""
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def build_qdrant_filter(filters: Optional[FilterParams]) -> Optional[Filter]:
    """
    Constructs a Qdrant Filter object.
    Qdrant applies these filters BEFORE the vector search (Pre-filtering),
    which is highly efficient.
    """
    if not filters:
        return None
    
    conditions = []

    # 1. Price Range Filter
    if filters.min_price is not None or filters.max_price is not None:
        conditions.append(
            FieldCondition(
                key="price",
                range=Range(
                    gte=filters.min_price,
                    lte=filters.max_price
                )
            )
        )

    # 2. Vendor Filter (Exact Match)
    if filters.vendor:
        conditions.append(
            FieldCondition(
                key="vendor",
                match=MatchValue(value=filters.vendor)
            )
        )

    # 3. Tags Filter (Match Any from list)
    # If user asks for "Blue", checks if "Blue" is in the 'tags' list payload
    if filters.allowed_tags:
        conditions.append(
            FieldCondition(
                key="tags",
                match=MatchAny(any=filters.allowed_tags)
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)

# --- API ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "Recommender System Online"}

@app.post("/search/semantic")
def semantic_search(request: SearchRequest):
    """
    🔍 Search by Meaning + Metadata Filters
    """
    query_vector = get_embedding(request.query)
    search_filter = build_qdrant_filter(request.filters)

    hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=search_filter,
        limit=request.limit
    )

    return {
        "query": request.query,
        "results": [format_hit(hit) for hit in hits]
    }

@app.post("/recommend/similar")
def recommend_similar(request: SimilarRequest):
    """
    👯 Item-to-Item Recommendation + Filters
    Changed to POST to allow complex filter body.
    """
    try:
        search_filter = build_qdrant_filter(request.filters)
        
        results = qdrant_client.recommend(
            collection_name=COLLECTION_NAME,
            positive=[request.product_id],
            query_filter=search_filter,
            limit=request.limit
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error: {str(e)}")

    return {
        "source_product": request.product_id,
        "recommendations": [format_hit(hit) for hit in results]
    }

@app.post("/recommend/personalized")
def personalized_recommendation(request: RecommendationRequest):
    """
    🧠 Contextual Recommendation + Filters
    (Vector(Liked) - Vector(Disliked)) + Filters
    """
    try:
        search_filter = build_qdrant_filter(request.filters)

        results = qdrant_client.recommend(
            collection_name=COLLECTION_NAME,
            positive=request.positive_product_ids,
            negative=request.negative_product_ids,
            query_filter=search_filter,
            limit=request.limit
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

    return {
        "context": {
            "liked": request.positive_product_ids,
            "disliked": request.negative_product_ids
        },
        "recommendations": [format_hit(hit) for hit in results]
    }

def format_hit(hit):
    return {
        "id": hit.id,
        "score": hit.score,
        "title": hit.payload.get("title"),
        "price": hit.payload.get("price"),
        "vendor": hit.payload.get("vendor"),
        "tags": hit.payload.get("tags")
    }

if __name__ == "__main__":
    print("🚀 Starting Advanced Recommender Engine on Port 8001...")
    uvicorn.run("recommender:app", host="0.0.0.0", port=8001, reload=True)