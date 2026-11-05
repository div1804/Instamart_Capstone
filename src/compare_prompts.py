
import pandas as pd
import mock_llm

from src.triage_prompts import ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT
from src.triage_retry import call_with_retry
from src.triage_validation import validate_response


# Configuration
INPUT_FILE = "outputs/redacted_tickets.csv"
OUTPUT_DIR = "outputs"

ALLOWED_STATUSES = {
    "classified_json",
    "classified_keyword"
}

CATEGORIES = [
    "billing",
    "delivery",
    "technical",
    "fraud"
]


def run_pipeline(prompt_name, system_prompt, token_budget=None):

    df = pd.read_csv(INPUT_FILE)

    # Reset immediately before each complete run
    mock_llm.reset()

    results = []
    total_tokens = 0

    for _, row in df.iterrows():

        ticket_id = row["ticket_id"]
        customer_id = row["customer_id"]
                # Read original and redacted ticket text
        raw_text = row["redacted_text"]
        redacted_text = row["redacted_text"]

        # Guardrail 1: Empty or whitespace text
        if pd.isna(raw_text) or not str(raw_text).strip():

            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "skipped_empty",
                "category": "",
                "true_category": row["true_category"],
                "tokens_used": 0
            })

            continue

        # Convert text after checking for empty values
        original_text = str(raw_text)

        # Guardrail 2: Long text
        if mock_llm.est_tokens(original_text) > 400:

            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "flagged_long",
                "category": "",
                "true_category": row["true_category"],
                "tokens_used": 0
            })

            continue
        # Optional budget check for Step 7
        if token_budget is not None and total_tokens >= token_budget:
            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "budget_exhausted",
                "category": "",
                "true_category": row["true_category"],
                "tokens_used": 0
            })
            continue

        # User message contains ONLY redacted ticket text
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

        # Resilient model call
        response = call_with_retry(ticket_id, messages)

        # Extract response and token usage
        response_text = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens

        total_tokens += tokens_used

        # Validate model response
        validation = validate_response(response_text)

        results.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": validation["status"],
            "category": validation["category"],
            "true_category": row["true_category"],
            "tokens_used": tokens_used
        })

    results_df = pd.DataFrame(results)

    # Calculate metrics only for classified tickets
    classified_df = results_df[
        results_df["status"].isin(ALLOWED_STATUSES)
    ].copy()

    classified_count = len(classified_df)

    if classified_count > 0:
        accuracy = (
            classified_df["category"]
            == classified_df["true_category"]
        ).mean() * 100

        tokens_per_ticket = total_tokens / classified_count
    else:
        accuracy = 0
        tokens_per_ticket = 0

    # Save individual results
    output_file = f"{OUTPUT_DIR}/{prompt_name}_results.csv"
    results_df.to_csv(output_file, index=False)

    summary = {
        "prompt": prompt_name,
        "classified_tickets": classified_count,
        "accuracy_percent": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_classified_ticket": tokens_per_ticket
    }

    print("\n" + "=" * 55)
    print(f"{prompt_name.upper()} RESULTS")
    print("=" * 55)

    print("Total tickets:", len(results_df))
    print("Classified tickets:", classified_count)
    print("JSON classifications:",
          (results_df["status"] == "classified_json").sum())
    print("Keyword classifications:",
          (results_df["status"] == "classified_keyword").sum())
    print("Invalid outputs:",
          (results_df["status"] == "flagged_invalid_output").sum())
    print("Skipped empty:",
          (results_df["status"] == "skipped_empty").sum())
    print("Flagged long:",
          (results_df["status"] == "flagged_long").sum())

    print(f"Accuracy: {accuracy:.2f}%")
    print("Total tokens:", total_tokens)
    print(f"Tokens per classified ticket: {tokens_per_ticket:.2f}")
    print("Saved:", output_file)

    return results_df, summary


# Run ZERO-SHOT without a token budget
zero_results, zero_summary = run_pipeline(
    "zero_shot",
    ZERO_SHOT_PROMPT
)

# Run FEW-SHOT without a token budget
few_results, few_summary = run_pipeline(
    "few_shot",
    FEW_SHOT_PROMPT
)


# Compare both approaches
comparison = pd.DataFrame([
    zero_summary,
    few_summary
])

print("\n\nPROMPT COMPARISON")
print("=" * 70)
print(comparison.to_string(index=False))


# Zero-shot confusion matrix
zero_classified = zero_results[
    zero_results["status"].isin(ALLOWED_STATUSES)
]

confusion = pd.crosstab(
    zero_classified["true_category"],
    zero_classified["category"]
)

confusion = confusion.reindex(
    index=CATEGORIES,
    columns=CATEGORIES,
    fill_value=0
)

print("\nZERO-SHOT CONFUSION MATRIX")
print("=" * 55)
print(confusion)

# Save comparison summary
comparison.to_csv(
    f"{OUTPUT_DIR}/prompt_comparison.csv",
    index=False
)

print("\nSTEP 6 COMPLETED!")
