import requests

class ShopifyGraphQLClient:
    def __init__(self):
        self.endpoint = f"https://{"aitken-store-2"}.myshopify.com/admin/api/2025-10/graphql.json"
        self.headers = {
            'Content-Type': 'application/json',
            'X-Shopify-Access-Token': "shpat_b0bd..."
        }
    
    def execute_query(self, query, variables=None):
        payload = {'query': query}
        if variables:
            payload['variables'] = variables
        
        response = requests.post(self.endpoint, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()
    


     
client = ShopifyGraphQLClient()
query = """
query {
  shop {
    id
    name
    email
    billingAddress {
      id
      address1
    }
  }
  products(first:10) {
    nodes {
      id
      title
      description
    }
  }
}
"""
client.execute_query(query)
 