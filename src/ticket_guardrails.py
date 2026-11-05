import pandas as pd

# 1. Load tickets
tickets = pd.read_csv("data/raw/tickets.csv")

# 2. Maximum input token limit
MAX_INPUT_TOKENS = 400

# 3. Token estimation function
def est_tokens(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())

# 4. Process tickets in original order
results = []

for _, ticket in tickets.iterrows():

    text = ticket["text"]

    if pd.isna(text) or str(text).strip() == "":
        status = "skipped_empty"

    elif est_tokens(text) > MAX_INPUT_TOKENS:
        status = "flagged_long"

    else:
        status = "eligible"

    results.append({
        "ticket_id": ticket["ticket_id"],
        "customer_id": ticket["customer_id"],
        "order_id": ticket["order_id"],
        "text": text,
        "true_category": ticket["true_category"],
        "estimated_tokens": est_tokens(text),
        "status": status
    })

# 5. Create result DataFrame
guardrail_results = pd.DataFrame(results)

# 6. Display status counts
print("INPUT GUARDRAIL RESULTS")
print("=" * 50)

print(guardrail_results["status"].value_counts())

print("\nTotal tickets:", len(guardrail_results))

print("\nEmpty tickets:")
print(
    guardrail_results[
        guardrail_results["status"] == "skipped_empty"
    ][["ticket_id", "status"]]
)

print("\nLong tickets:")
print(
    guardrail_results[
        guardrail_results["status"] == "flagged_long"
    ][["ticket_id", "estimated_tokens", "status"]]
)

print("\nEligible tickets:",
      (guardrail_results["status"] == "eligible").sum())

print("\nINPUT GUARDRAILS COMPLETED!")