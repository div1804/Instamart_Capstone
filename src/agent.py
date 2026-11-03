
import pandas as pd
import os

def build_scenarios():
    orders = pd.read_csv("data/clean/orders.csv")

    # Sort by order ID to make scenario selection deterministic
    orders = orders.sort_values("order_id")

    # Select valid rows
    valid = orders.dropna(
        subset=["order_id", "customer_id", "status", "order_amount"]
    ).copy()

    valid["delivery_minutes"] = pd.to_numeric(
        valid["delivery_minutes"], errors="coerce"
    )
    valid["order_amount"] = pd.to_numeric(
        valid["order_amount"], errors="coerce"
    )

    valid = valid.dropna(subset=["order_amount"])

    # S1 and S2: late delivery, amount <= INR 1000
    late_small = valid[
        (valid["status"] == "completed")
        & (valid["delivery_minutes"] > 45)
        & (valid["order_amount"] <= 1000)
    ].head(2)

    # S3: late delivery, amount > INR 1000
    late_big = valid[
        (valid["status"] == "completed")
        & (valid["delivery_minutes"] > 45)
        & (valid["order_amount"] > 1000)
    ].iloc[0]

    # S4: completed, delivered within 45 minutes
    on_time = valid[
        (valid["status"] == "completed")
        & (valid["delivery_minutes"] <= 45)
    ].iloc[0]

    # S5: cancelled order
    cancelled = valid[
        valid["status"] == "cancelled"
    ].iloc[0]

    scenarios = []

    for _, row in late_small.iterrows():
        scenarios.append({
            "customer_id": row["customer_id"],
            "message": (
                f"My order {row['order_id']} reached me very late. "
                "I would like a refund."
            ),
            "expected_outcome": "REFUND",
            "expected_refund_inr": float(row["order_amount"])
        })

    scenarios.append({
        "customer_id": late_big["customer_id"],
        "message": (
            f"Order {late_big['order_id']} arrived far too late, "
            "please refund me."
        ),
        "expected_outcome": "ESCALATE",
        "expected_refund_inr": float(late_big["order_amount"])
    })

    scenarios.append({
        "customer_id": on_time["customer_id"],
        "message": (
            f"Please refund order {on_time['order_id']}, "
            "I did not like the delivery."
        ),
        "expected_outcome": "DENIED",
        "expected_refund_inr": 0
    })

    scenarios.append({
        "customer_id": cancelled["customer_id"],
        "message": (
            f"I want my money back for order {cancelled['order_id']}."
        ),
        "expected_outcome": "DENIED",
        "expected_refund_inr": 0
    })

    scenarios.append({
        "customer_id": "C0001",
        "message": "Refund for order ORD-999999 please.",
        "expected_outcome": "NOT_FOUND",
        "expected_refund_inr": 0
    })

    first_late = late_small.iloc[0]

    scenarios.append({
        "customer_id": first_late["customer_id"],
        "message": f"[LOOP] Refund order {first_late['order_id']}.",
        "expected_outcome": "FALLBACK",
        "expected_refund_inr": 0
    })

    for i, scenario in enumerate(scenarios, start=1):
        scenario["scenario_id"] = f"S{i}"

    os.makedirs("data/raw", exist_ok=True)

    result = pd.DataFrame(scenarios)

    result = result[
        [
            "scenario_id",
            "customer_id",
            "message",
            "expected_outcome",
            "expected_refund_inr"
        ]
    ]

    result.to_csv("data/raw/agent_scenarios.csv", index=False)

    return result
