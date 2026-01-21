import os
import json
import requests
from openai import OpenAI
import time
from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURATION ---
# Replace these with your actual keys or set them as environment variables
SHOPIFY_ACCESS_TOKEN =os.getenv("SHOPIFY_ACCESS_TOKEN")
SHOPIFY_SHOP_URL = os.getenv("SHOPIFY_SHOP_URL")
STORE_CONTEXT = "Consumer Electronics and Gadgets" 
NUM_PRODUCTS_TO_GENERATE = 1

# ---------------------



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_product_with_gpt4(niche):
    """
    Asks GPT-4o to generate a realistic product JSON object for ANY niche.
    """
    print(f"🤖 Asking GPT-4o for a new '{niche}' product...")

    prompt = f"""
    Generate a detailed, realistic product for a Shopify store selling {niche}.
    Return ONLY a valid JSON object. Do not include markdown formatting.
    
    The JSON must have this exact structure:
    {{
        "title": "Product Name",
        "body_html": "<p>Engaging description...</p>",
        "vendor": "Brand Name",
        "product_type": "The simple category name (e.g. 'Smartphone', 'T-Shirt', 'Sofa')",
        "price": "149.99",
        "tags": "tag1, tag2, tag3",
        "sku": "SKU-123",
        "taxonomy_id": "gid://shopify/TaxonomyCategory/...", 
        "description": "...."
        "attributes": {{
            "Attribute Name": "Value"
        }}
    }}

    NOTE ON TAXONOMY_ID: 
    You MUST provide the correct Shopify Taxonomy Category GID.
    - For Electronics, IDs often start with 'el-' (e.g., gid://shopify/TaxonomyCategory/el-123).
    - For Apparel, IDs often start with 'aa-'.
    - For Home, IDs often start with 'hg-'.
    Select the most specific Leaf Category possible for the item.

    NOTE ON ATTRIBUTES: 
    Generate 3-5 specific technical specifications relevant to the item.
    - Electronics: 'Screen Size', 'Battery Life', 'Processor', 'Connectivity'.
    - Apparel: 'Material', 'Fit', 'Care Instructions'.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert Shopify e-commerce manager capable of handling any product category."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"❌ Error generating product with GPT: {e}")
        return None

def update_product_category_graphql(product_id, taxonomy_gid):
    """
    Updates the product's Standardized Category (Taxonomy) using GraphQL.
    """
    url = f"https://{SHOPIFY_SHOP_URL}/admin/api/2024-10/graphql.json"
    headers = {
        "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN,
        "Content-Type": "application/json"
    }

    mutation = """
    mutation productUpdate($input: ProductInput!) {
      productUpdate(input: $input) {
        product {
          id
          category {
            id
            name
          }
        }
        userErrors {
          field
          message
        }
      }
    }
    """

    variables = {
        "input": {
            "id": f"gid://shopify/Product/{product_id}",
            "category": taxonomy_gid
        }
    }

    try:
        response = requests.post(url, json={"query": mutation, "variables": variables}, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        if 'data' in result and 'productUpdate' in result['data']:
            user_errors = result['data']['productUpdate']['userErrors']
            if user_errors:
                print(f"   ⚠️ GraphQL Error setting Category: {user_errors}")
                print(f"      (Tried to set: {taxonomy_gid})")
            else:
                cat_name = result['data']['productUpdate']['product']['category']['name']
                print(f"   -> ✅ Taxonomy Category set to: {cat_name}")
        else:
             print(f"   ⚠️ Unexpected GraphQL response: {result}")

    except Exception as e:
        print(f"   ❌ GraphQL Request Failed: {e}")

def create_shopify_product(product_data):
    """
    Uploads the product via REST, then updates Category via GraphQL.
    """
    url = f"https://{SHOPIFY_SHOP_URL}/admin/api/2024-10/products.json"
    headers = {
        "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN,
        "Content-Type": "application/json"
    }

    # Process Metafields (Generic handling)
    metafields_payload = []
    if "attributes" in product_data:
        for key, value in product_data["attributes"].items():
            safe_key = key.lower().replace(" ", "_").replace("-", "_")
            # Limit key length to 30 chars to match Shopify limits
            safe_key = safe_key[:30]
            
            metafields_payload.append({
                "namespace": "custom",
                "key": safe_key,
                "value": str(value), # Ensure value is string
                "type": "single_line_text_field",
                "description": key
            })

    payload = {
        "product": {
            "title": product_data["title"],
            "body_html": product_data["body_html"],
            "vendor": product_data["vendor"],
            "product_type": product_data["product_type"],
            "tags": product_data["tags"],
            "status": "active",
            "variants": [
                {
                    "price": product_data["price"],
                    "sku": product_data["sku"],
                    "inventory_management": "shopify",
                    "inventory_quantity": 50
                }
            ],
            "metafields": metafields_payload
        }
    }

    try:
        # Step 1: Create Product via REST
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        new_product = response.json()['product']
        print(f"✅ Created: {new_product['title']} | ID: {new_product['id']}")

        # Step 2: Update Category via GraphQL
        if "taxonomy_id" in product_data and product_data["taxonomy_id"]:
            update_product_category_graphql(new_product['id'], product_data["taxonomy_id"])
        else:
            print("   ⚠️ No Taxonomy ID generated by GPT. Skipping category assignment.")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Shopify REST API Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Optional: Allow user to override context at runtime
    print("--- Shopify Generic Store Populator ---")
    user_context = input(f"Enter Store Niche (Press Enter for '{STORE_CONTEXT}'): ")
    if user_context.strip():
        STORE_CONTEXT = user_context.strip()

    print(f"\n🚀 Starting generation of {NUM_PRODUCTS_TO_GENERATE} products for: '{STORE_CONTEXT}'...\n")
    
    for i in range(NUM_PRODUCTS_TO_GENERATE):
        print(f"--- Product {i+1}/{NUM_PRODUCTS_TO_GENERATE} ---")
        
        product_data = generate_product_with_gpt4(STORE_CONTEXT)
        
        if product_data:
            create_shopify_product(product_data)
            time.sleep(1.5)
        
    print("\n✨ All done!")