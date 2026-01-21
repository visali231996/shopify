# Shopify +Qdarnt chatbot agent
This uses shopify website and performs additions, deletes,modifies items into the website..we can also add items to the cart
Sales bot can create products in Shopify (Admin API).
Shopify webhooks push create/update/delete events to a listener.
The listener syncs Qdrant (upsert/delete). On updates it stores a small reflection (change_diff, change_summary).

## .env file contains
OPENAI_API_KEY
SHOPIFY_STORE_DOMAIN
SHOPIFY_ADMIN_ACCESS_TOKEN
SHOPIFY_STOREFRONT_TOKEN
SHOPIFY_WEBHOOK_SECRET

## Running the agent
1. start Qdrant
```
docker run -p 6333:6333 qdrant/qdrant
```
2. Start Webhook Listener
```
python shopify_webhook.py
```
3. Start sales bot 
```
streamlit run codefinal.py
```
4. port using ngrok
```
ngrok http 8001
```
## Shopify Webhooks
Create Shopify webhooks to point to:

products/create -> http://<your-public-host>/webhooks/shopify/products-create
products/update -> http://<your-public-host>/webhooks/shopify/products-update
products/delete -> http://<your-public-host>/webhooks/shopify/products-deletion