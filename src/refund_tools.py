
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Retrieve order details using an order ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Unique order ID"
                }
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search policy documents using a customer query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Policy search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the refund amount based on order amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {
                    "type": "number",
                    "description": "Total order amount in INR"
                },
                "delivery_minutes": {
                    "type": "integer",
                    "description": "Delivery time in minutes"
                }
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Record an approved simulated refund.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string"
                },
                "refund_inr": {
                    "type": "number"
                }
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund request that requires human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string"
                },
                "refund_inr": {
                    "type": "number"
                }
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]

print("Number of tools:", len(TOOLS_SPEC))

for tool in TOOLS_SPEC:
    print(tool["name"])
    
import sqlite3
import json
import os
from datetime import datetime

from policy_retrieval import retrieve


DB_PATH = "data/instamart.db"
LEDGER_PATH = "outputs/refund_ledger.jsonl"


# TOOL 1: Get order details
def get_order(order_id):

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT order_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE order_id = ?
        """,
        (order_id,)
    )

    row = cursor.fetchone()
    connection.close()

    if row is None:
        return {"error": "order not found"}

    return dict(row)


# TOOL 2: Search policy
def search_policy(query):

    results = retrieve(query, k=3)

    return {"results": results}


# TOOL 3: Calculate refund
def calculate_refund(order_amount, delivery_minutes):

    if delivery_minutes > 45:
        refund_inr = order_amount
    else:
        refund_inr = 0

    return {"refund_inr": round(float(refund_inr), 2)}


# Helper function for refund ledger
def append_ledger(tool_name, order_id, refund_inr):

    os.makedirs("outputs", exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": refund_inr
    }

    with open(LEDGER_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


# TOOL 4: Issue refund
def issue_refund(order_id, refund_inr):

    append_ledger("issue_refund", order_id, refund_inr)

    return {"refund_inr": refund_inr}


# TOOL 5: Escalate to human
def escalate_to_human(order_id, refund_inr):

    append_ledger("escalate_to_human", order_id, refund_inr)

    return {"refund_inr": refund_inr}


# Map tool names to functions
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


print("TOOL_MAP created successfully!")
print("Available tools:", list(TOOL_MAP.keys()))

