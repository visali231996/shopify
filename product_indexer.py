import argparse
import uvicorn
import asyncio
from typing import List, Dict, Optional
from fastapi import FastAPI, Request, HTTPException
from langchain_openai import OpenAIEmbeddings
from tools.shopify_client import shopify_client
from memory.db_managers import qdrant_db
from config.settings import settings

# Initialize Embeddings
# Ensure OPENAI_API_KEY is set in settings/.env
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    api_key=settings.OPENAI_API_KEY
)

app = FastAPI(title="Shopify Product Webhook Listener")

class ProductIndexer:
    def __init__(self):
        self.collection_name = "shopify_products"

    async def generate_embedding(self, text: str) -> List[float]:
        return await embeddings_model.aembed_query(text)

    async def index_product(self, product_data: Dict):
        """
        Processes a single product dictionary (from GraphQL or Webhook)
        and inserts it into Qdrant.
        """
        # Handle structure differences between GraphQL and Webhook JSON
        p_id = str(product_data.get("id", ""))
        title = product_data.get("title", "")
        # GraphQL uses 'description', Webhooks might use 'body_html' or 'body'
        desc = product_data.get("description") or product_data.get("body_html") or ""
        
        # Extract Price (Simplified)
        price = "0.00"
        if "variants" in product_data:
            # GraphQL structure
            variants = product_data["variants"]
            if isinstance(variants, dict) and "edges" in variants:
                 if variants["edges"]:
                     price = variants["edges"][0]["node"].get("price", "0.00")
            # Webhook structure (list of dicts)
            elif isinstance(variants, list) and len(variants) > 0:
                price = variants[0].get("price", "0.00")

        text_to_embed = f"Product: {title}. Description: {desc}. Price: {price}"
        
        print(f"Generate embedding for: {title}")
        vector = await self.generate_embedding(text_to_embed)

        payload = {
            "product_id": p_id,
            "title": title,
            "description": desc,
            "price": price,
            "raw_text": text_to_embed
        }

        qdrant_db.upsert_point(self.collection_name, p_id, vector, payload)
        print(f"Indexed product: {title} ({p_id})")

    async def sync_all_products(self):
        print("--- Starting Bulk Sync from Shopify ---")
        cursor = None
        has_next = True
        
        while has_next:
            query = """
            query ($cursor: String) {
              products(first: 10, after: $cursor) {
                pageInfo {
                  hasNextPage
                  endCursor
                }
                edges {
                  node {
                    id
                    title
                    description
                    variants(first: 1) {
                        edges {
                            node {
                                price
                            }
                        }
                    }
                  }
                }
              }
            }
            """
            
            result = shopify_client.execute(query, {"cursor": cursor})
            data = result.get("data", {}).get("products", {})
            
            edges = data.get("edges", [])
            for edge in edges:
                node = edge["node"]
                await self.index_product(node)
            
            page_info = data.get("pageInfo", {})
            has_next = page_info.get("hasNextPage", False)
            cursor = page_info.get("endCursor")
            
        print("--- Bulk Sync Complete ---")

indexer = ProductIndexer()

# --- Webhook Endpoints ---

@app.post("/webhooks/shopify/product-update")
async def product_update_webhook(request: Request):
    """
    Endpoint for 'products/create' and 'products/update' webhooks.
    """
    # Verify HMAC here in production using settings.SHOPIFY_WEBHOOK_SECRET
    
    try:
        payload = await request.json()
        # Webhook payload is usually the product object directly
        await indexer.index_product(payload)
        return {"status": "success", "message": "Product indexed"}
    except Exception as e:
        print(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

# --- Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shopify Product Indexer & Webhook Server")
    parser.add_argument("--sync", action="store_true", help="Run bulk sync of all Shopify products")
    parser.add_argument("--server", action="store_true", help="Start the Webhook Server")
    
    args = parser.parse_args()

    if args.sync:
        asyncio.run(indexer.sync_all_products())
    elif args.server:
        print("Starting Webhook Server on port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Please specify --sync or --server")