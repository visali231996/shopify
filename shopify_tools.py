import httpx
import asyncio
import logging
from typing import Dict, Any, Optional, List

# Configure module-level logger
logger = logging.getLogger("shopify_tools")

class ShopifyClient:
    """
    Async client for Shopify Admin API (GraphQL) with rate limit handling.
    """
    def __init__(self, shop_url: str, access_token: str, api_version: str = "2024-01"):
        self.shop_url = shop_url.replace("https://", "").replace("/", "")
        self.access_token = access_token
        self.api_version = api_version
        self.base_url = f"https://{self.shop_url}/admin/api/{self.api_version}/graphql.json"
        self.headers = {
            "X-Shopify-Access-Token": self.access_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _make_request(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Internal method to execute GraphQL requests with 429 (Too Many Requests) handling.
        """
        payload = {"query": query, "variables": variables or {}}
        
        async with httpx.AsyncClient() as client:
            retries = 3
            for attempt in range(retries):
                try:
                    response = await client.post(self.base_url, headers=self.headers, json=payload, timeout=10.0)
                    
                    # Handle Rate Limiting
                    if response.status_code == 429:
                        retry_after = float(response.headers.get("Retry-After", 2.0))
                        logger.warning(f"Rate limit hit. Retrying in {retry_after} seconds...")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
                    json_res = response.json()
                    
                    # Check for GraphQL-level errors (which return 200 OK but contain 'errors' key)
                    if "errors" in json_res:
                        logger.error(f"GraphQL Errors: {json_res['errors']}")
                        raise Exception(f"GraphQL Error: {json_res['errors'][0]['message']}")
                        
                    return json_res

                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP Error: {e.response.text}")
                    raise e
                except httpx.RequestError as e:
                    logger.error(f"Request Error: {e}")
                    raise e
            
            raise Exception("Max retries exceeded for Shopify API request")

    async def execute_query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self._make_request(query, variables)

    async def execute_mutation(self, mutation: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self._make_request(mutation, variables)

    # ---------------------------------------------------------
    # SPECIFIC TOOLS
    # ---------------------------------------------------------

    async def get_products(self, limit: int = 10, query: str = "") -> Dict[str, Any]:
        gql = """
        query getProducts($first: Int!, $query: String) {
          products(first: $first, query: $query) {
            edges {
              node {
                id
                title
                description
                totalInventory
                priceRangeV2 {
                  minVariantPrice { amount currencyCode }
                }
                variants(first: 5) {
                  edges {
                    node { id title inventoryQuantity price }
                  }
                }
              }
            }
          }
        }
        """
        return await self._make_request(gql, {"first": limit, "query": query})

    async def get_customer_by_email(self, email: str) -> Dict[str, Any]:
        gql = """
        query getCustomerByEmail($query: String!) {
          customers(first: 1, query: $query) {
            edges {
              node {
                id
                firstName
                lastName
                email
                lifetimeDuration
                amountSpent { amount currencyCode }
                ordersCount
              }
            }
          }
        }
        """
        response = await self._make_request(gql, {"query": f"email:{email}"})
        edges = response.get("data", {}).get("customers", {}).get("edges", [])
        return edges[0]["node"] if edges else None

    async def get_customer_orders(self, customer_id: str, limit: int = 5) -> Dict[str, Any]:
        gql = """
        query getCustomerOrders($id: ID!, $first: Int!) {
          customer(id: $id) {
            orders(first: $first, sortKey: PROCESSED_AT, reverse: true) {
              edges {
                node {
                  id
                  name
                  processedAt
                  financialStatus
                  totalPriceSet { shopMoney { amount currencyCode } }
                  lineItems(first: 5) {
                    edges { node { title quantity } }
                  }
                }
              }
            }
          }
        }
        """
        return await self._make_request(gql, {"id": customer_id, "first": limit})

    async def create_discount(self, code: str, amount: float, is_percentage: bool = True) -> Dict[str, Any]:
        """
        Creates a basic discount code.
        """
        gql = """
        mutation discountCodeBasicCreate($basicCodeDiscount: DiscountCodeBasicInput!) {
          discountCodeBasicCreate(basicCodeDiscount: $basicCodeDiscount) {
            codeDiscountNode {
              codeDiscount {
                ... on DiscountCodeBasic {
                  title
                  codes(first: 1) { edges { node { code } } }
                  status
                }
              }
            }
            userErrors { field message }
          }
        }
        """
        
        # Determine value type
        value_input = {"percentage": amount / 100.0} if is_percentage else {"amount": amount}
        
        variables = {
            "basicCodeDiscount": {
                "title": f"Agent Generated: {code}",
                "code": code,
                "startsAt": "2024-01-01T00:00:00Z", # In real app, use current time
                "customerSelection": {"all": True},
                "customerGets": {
                    "value": value_input,
                    "items": {"all": True}
                }
            }
        }
        return await self._make_request(gql, variables)

    async def get_active_discounts(self, limit: int = 10) -> Dict[str, Any]:
        gql = """
        query getDiscounts($first: Int!) {
          discountNodes(first: $first, query: "status:ACTIVE") {
            edges {
              node {
                id
                discount {
                  ... on DiscountCodeBasic {
                    title
                    codes(first: 1) { edges { node { code } } }
                    summary
                  }
                }
              }
            }
          }
        }
        """
        return await self._make_request(gql, {"first": limit})

    async def create_checkout_url(self, variant_id: str, quantity: int, customer_email: Optional[str] = None) -> str:
        """
        Creates a Draft Order to act as an instant checkout link.
        """
        gql = """
        mutation draftOrderCreate($input: DraftOrderInput!) {
          draftOrderCreate(input: $input) {
            draftOrder {
              id
              invoiceUrl
            }
            userErrors { field message }
          }
        }
        """
        variables = {
            "input": {
                "lineItems": [{"variantId": variant_id, "quantity": quantity}],
                "email": customer_email
            }
        }
        res = await self._make_request(gql, variables)
        data = res.get("data", {}).get("draftOrderCreate", {})
        
        if data.get("userErrors"):
            raise Exception(f"Checkout Creation Failed: {data['userErrors']}")
            
        return data.get("draftOrder", {}).get("invoiceUrl")
    
    async def create_order(self, variant_id: str, quantity: int, customer_email: str, note: str = "Created via API") -> Dict[str, Any]:
        """
        Creates a real Order by first creating a Draft Order and then completing it.
        This marks the order as 'Pending' payment by default unless payment is captured.
        """
        
        # Step 1: Create Draft Order
        draft_gql = """
        mutation draftOrderCreate($input: DraftOrderInput!) {
          draftOrderCreate(input: $input) {
            draftOrder { id }
            userErrors { field message }
          }
        }
        """
        draft_vars = {
            "input": {
                "lineItems": [{"variantId": variant_id, "quantity": quantity}],
                "email": customer_email,
                "note": note,
                "tags": ["api-generated"]
            }
        }
        
        draft_res = await self._make_request(draft_gql, draft_vars)
        draft_data = draft_res.get("data", {}).get("draftOrderCreate", {})
        
        if draft_data.get("userErrors"):
             raise Exception(f"Draft Creation Failed: {draft_data['userErrors']}")
        
        draft_id = draft_data.get("draftOrder", {}).get("id")
        
        # Step 2: Complete Draft Order (transitions to Real Order)
        complete_gql = """
        mutation draftOrderComplete($id: ID!) {
          draftOrderComplete(id: $id) {
            draftOrder {
              order {
                id
                name
                totalPriceSet { shopMoney { amount currencyCode } }
              }
            }
            userErrors { field message }
          }
        }
        """
        
        complete_res = await self._make_request(complete_gql, {"id": draft_id})
        complete_data = complete_res.get("data", {}).get("draftOrderComplete", {})
        
        if complete_data.get("userErrors"):
            raise Exception(f"Order Completion Failed: {complete_data['userErrors']}")
            
        return complete_data.get("draftOrder", {}).get("order")

    async def get_inventory(self, variant_ids: List[str]) -> Dict[str, Any]:
        # Retrieving specific variants to check levels
        gql = """
        query getInventory($ids: [ID!]!) {
          nodes(ids: $ids) {
            ... on ProductVariant {
              id
              title
              inventoryQuantity
              inventoryItem {
                tracked
              }
            }
          }
        }
        """
        return await self._make_request(gql, {"ids": variant_ids})

    async def get_shop_insights(self) -> Dict[str, Any]:
        gql = """
        query getShop {
          shop {
            name
            currencyCode
            email
            myshopifyDomain
            description
          }
        }
        """
        return await self._make_request(gql)