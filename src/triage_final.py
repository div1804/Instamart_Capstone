
import pandas as pd
import mock_llm

from src.triage_prompts import ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT
from src.triage_retry import call_with_retry
from src.triage_validation import validate_response


# File paths
TICKETS_FILE = "outputs/redacted_tickets.csv"
SEGMENTS_FILE = "outputs/customer_segments.csv"

# Allowed successful statuses
CLASSIFIED_STATUSES = {
    "classified_json",
    "classified_keyword"
}

HUMAN_REVIEW_STATUSES = {
    "skipped_empty",
    "flagged_long",
    "flagged_invalid_output",
    "budget_exhausted"
}


def run_ticket_pipeline(system_prompt, token_budget):

    tickets = pd.read_csv(TICKETS_FILE)

    # Reset immediately before each complete run
    mock_llm.reset()

    results = []
    total_tokens = 0

    for _, row in tickets.iterrows():

        ticket_id = row["ticket_id"]
        customer_id = row["customer_id"]

        raw_text = row["text"]
        redacted_text = row["redacted_text"]

        # Guardrail 1: Empty text
        if pd.isna(raw_text) or not str(raw_text).strip():

            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "skipped_empty",
                "category": "",
                "tokens_used": 0
            })

            continue

        original_text = str(raw_text)

        # Guardrail 2: Long text
        if mock_llm.est_tokens(original_text) > 400:

            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "flagged_long",
                "category": "",
                "tokens_used": 0
            })

            continue

        # Budget check before each model call
        if total_tokens >= token_budget:

            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "budget_exhausted",
                "category": "",
                "tokens_used": 0
            })

            continue

        # User content must be ONLY the redacted ticket text
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": redacted_text
            }
        ]

        # Call model with retry handling
        response = call_with_retry(ticket_id, messages)

        response_text = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens

        total_tokens += tokens_used

        # Validate model output
        validation = validate_response(response_text)

        results.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": validation["status"],
            "category": validation["category"],
            "tokens_used": tokens_used
        })

    return pd.DataFrame(results), total_tokens


# =====================================================
# RUN 1: ZERO-SHOT WITH 6000 TOKEN BUDGET
# =====================================================

print("\nZERO-SHOT BUDGET DEMONSTRATION")
print("=" * 55)

budget_results, budget_used = run_ticket_pipeline(
    ZERO_SHOT_PROMPT,
    token_budget=6000
)

budget_results.to_csv(
    "outputs/triage_budget_demo.csv",
    index=False
)

print("\nBudget demonstration saved!")
print("File: outputs/triage_budget_demo.csv")
print("Token budget: 6000")
print("Tokens used:", budget_used)

print("\nBudget run status counts:")
print(budget_results["status"].value_counts())

print(
    "\nBudget exhausted tickets:",
    (budget_results["status"] == "budget_exhausted").sum()
)


# =====================================================
# RUN 2: FEW-SHOT WITH 100000 TOKEN BUDGET
# =====================================================

print("\n\nFINAL FEW-SHOT TICKET CLASSIFICATION")
print("=" * 55)

final_results, final_tokens = run_ticket_pipeline(
    FEW_SHOT_PROMPT,
    token_budget=100000
)

print("\nFinal model run completed.")
print("Total tokens used:", final_tokens)


# =====================================================
# CUSTOMER SEGMENT INFORMATION
# =====================================================

segments = pd.read_csv(SEGMENTS_FILE)

# Keep required customer segment columns
segments = segments[
    ["customer_id", "segment_name", "is_premium"]
].copy()

# Join ticket results with customer segments
final_results = final_results.merge(
    segments,
    on="customer_id",
    how="left"
)

# Missing customers are treated as non-premium
final_results["is_premium"] = (
    final_results["is_premium"]
    .fillna(0)
    .astype(int)
)

final_results["segment_name"] = (
    final_results["segment_name"]
    .fillna("")
)


# =====================================================
# ROUTING AND PRIORITY RULES
# =====================================================

def assign_routing(row):

    status = row["status"]
    category = row["category"]
    is_premium = row["is_premium"]

    # Human review cases
    if status in HUMAN_REVIEW_STATUSES:
        return pd.Series({
            "route": "human_review",
            "priority": "NORMAL"
        })

    # Fraud tickets
    if category == "fraud":
        return pd.Series({
            "route": "security_team",
            "priority": "HIGH"
        })

    # Premium customers
    if is_premium == 1:
        return pd.Series({
            "route": "premium_desk",
            "priority": "HIGH"
        })

    # Remaining tickets
    return pd.Series({
        "route": "support_queue",
        "priority": "NORMAL"
    })


routing = final_results.apply(
    assign_routing,
    axis=1
)

final_results["route"] = routing["route"]
final_results["priority"] = routing["priority"]


# =====================================================
# FINAL OUTPUT WITH EXACT REQUIRED COLUMNS
# =====================================================

required_columns = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used"
]

final_output = final_results[required_columns].copy()

final_output.to_csv(
    "outputs/triage_results.csv",
    index=False
)


# =====================================================
# FINAL VALIDATION
# =====================================================

print("\nFINAL TRIAGE RESULTS")
print("=" * 55)

print("Total tickets:", len(final_output))

print("\nStatus counts:")
print(final_output["status"].value_counts())

print("\nRouting counts:")
print(final_output["route"].value_counts())

print("\nPriority counts:")
print(final_output["priority"].value_counts())

print("\nFinal columns:")
print(final_output.columns.tolist())

print("\nFirst five final records:")
print(final_output.head().to_string(index=False))

print("\nTotal few-shot tokens:", final_tokens)

print("\nFINAL FILE SAVED SUCCESSFULLY!")
print("File: outputs/triage_results.csv")

print("\nPART C STEP 7 COMPLETED!")
