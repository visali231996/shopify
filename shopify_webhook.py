import uvicorn
import hmac
import hashlib
import base64
import json
import os
from fastapi import FastAPI, Request, Header, HTTPException, BackgroundTasks
from bs4 import BeautifulSoup
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList
from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURATION ---
# NOTE: Replace these with your actual environment variables or secure secrets management
SHOPIFY_SECRET = os.getenv("SHOPIFY_WEBHOOK_SECRET", "your_shopify_secret_here")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_key_here")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333") 
COLLECTION_NAME = "shopify_products"

# --- INITIALIZE CLIENTS ---
app = FastAPI()
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL)

# --- UTILITIES ---

async def verify_shopify_hmac(request: Request, x_shopify_hmac_sha256: str) -> bytes:
    """
    Verifies the Shopify HMAC signature. 
    Returns the raw body bytes if valid, raises HTTPException if invalid.
    """
    body_bytes = await request.body()
    try:
        digest = hmac.new(
            SHOPIFY_SECRET.encode('utf-8'),
            body_bytes,
            hashlib.sha256
        ).digest()
        computed_hmac = base64.b64encode(digest).decode('utf-8')
    except Exception:
        raise HTTPException(status_code=500, detail="Crypto error")

    if not x_shopify_hmac_sha256 or not hmac.compare_digest(computed_hmac, x_shopify_hmac_sha256):
        print("⚠️ Invalid Signature")
        raise HTTPException(status_code=401, detail="Invalid Signature")
    
    return body_bytes

@app.on_event("startup")
def startup_event():
    """
    Ensure the Qdrant collection exists on startup.
    """
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"Creating Qdrant collection: {COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

# --- BACKGROUND TASKS ---

def process_and_ingest_product(product_data: dict):
    """
    Used for both CREATE and UPDATE events.
    Qdrant 'upsert' will overwrite the existing point if the ID matches.
    """
    try:
        product_id = product_data.get("id")
        title = product_data.get("title", "")
        raw_html = product_data.get("body_html") or ""
        vendor = product_data.get("vendor", "")
        tags = product_data.get("tags", "")
        handle = product_data.get("handle", "")
        
        # Handle price safely (some products might not have variants or price)
        variants = product_data.get("variants", [])
        price = variants[0].get("price") if variants else "0.00"
        
        # 1. Clean HTML
        soup = BeautifulSoup(raw_html, "html.parser")
        clean_description = soup.get_text(separator=" ")
        
        # 2. Prepare Text for Embedding
        text_to_embed = f"Product: {title}. Vendor: {vendor}. Tags: {tags}. Description: {clean_description}"
        
        print(f"🔄 Upserting (Create/Update) Product ID: {product_id}...")
        
        # 3. Generate Embedding
        response = openai_client.embeddings.create(
            input=text_to_embed,
            model="text-embedding-3-small"
        )
        embedding_vector = response.data[0].embedding

        # 4. Upsert into Qdrant
        payload = {
            "title": title,
            "vendor": vendor,
            "price": price,
            "handle": handle,
            "tags": tags
        }

        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=product_id, 
                    vector=embedding_vector,
                    payload=payload
                )
            ]
        )
        print(f"✅ Successfully Upserted Product {product_id}")

    except Exception as e:
        print(f"❌ Upsert Task Failed: {e}")

def delete_product_from_qdrant(product_id: int):
    """
    Used for DELETE events.
    Removes the point from Qdrant based on the Product ID.
    """
    try:
        print(f"🗑️ Deleting Product ID: {product_id} from Qdrant...")
        
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(
                points=[product_id]
            )
        )
        print(f"✅ Successfully Deleted Product {product_id}")
        
    except Exception as e:
        print(f"❌ Delete Task Failed: {e}")

# --- ROUTES ---

@app.get("/")
async def health_check():
    return {"status": "active", "message": "Shopify Vector Sync is running"}

@app.post("/webhooks/shopify/products-create")
async def handle_product_create(
    request: Request, 
    background_tasks: BackgroundTasks,
    x_shopify_hmac_sha256: str = Header(None)
):
    body_bytes = await verify_shopify_hmac(request, x_shopify_hmac_sha256)
    product_data = json.loads(body_bytes)
    
    # Ingest (Create)
    background_tasks.add_task(process_and_ingest_product, product_data)
    return {"status": "received"}

@app.post("/webhooks/shopify/products-update")
async def handle_product_update(
    request: Request, 
    background_tasks: BackgroundTasks,
    x_shopify_hmac_sha256: str = Header(None)
):
    body_bytes = await verify_shopify_hmac(request, x_shopify_hmac_sha256)
    product_data = json.loads(body_bytes)
    
    # Ingest (Update - Overwrites existing ID)
    background_tasks.add_task(process_and_ingest_product, product_data)
    return {"status": "received"}

@app.post("/webhooks/shopify/products-deletion")
async def handle_product_delete(
    request: Request, 
    background_tasks: BackgroundTasks,
    x_shopify_hmac_sha256: str = Header(None)
):
    body_bytes = await verify_shopify_hmac(request, x_shopify_hmac_sha256)
    data = json.loads(body_bytes)
    
    # The delete payload is smaller, usually just {"id": 12345...}
    product_id = data.get("id")
    
    if product_id:
        background_tasks.add_task(delete_product_from_qdrant, product_id)
        
    return {"status": "received"}

if __name__ == "__main__":
    print("🚀 Starting Webhook Listener...")
    uvicorn.run("shopify_webhook:app", host="0.0.0.0", port=8000, reload=True)