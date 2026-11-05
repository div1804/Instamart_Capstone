import pandas as pd
import re

# 1. Load tickets
tickets = pd.read_csv("data/raw/tickets.csv")

# 2. Define exact patterns
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"ORD-\d{6}"

# 3. Create a copy of ticket data
redacted_tickets = tickets.copy()

# 4. Initialize substitution counters
email_count = 0
phone_count = 0
order_count = 0

# 5. Redact personal information
def redact_text(text):

    global email_count, phone_count, order_count

    if pd.isna(text):
        return text

    text = str(text)

    # Email redaction
    text, emails = re.subn(
        EMAIL_PATTERN,
        "[EMAIL]",
        text
    )
    email_count += emails

    # Phone redaction
    text, phones = re.subn(
        PHONE_PATTERN,
        "[PHONE]",
        text
    )
    phone_count += phones

    # Order ID redaction
    text, orders = re.subn(
        ORDER_PATTERN,
        "[ORDER_ID]",
        text
    )
    order_count += orders

    return text


# 6. Apply redaction to all tickets
redacted_tickets["redacted_text"] = (
    redacted_tickets["text"].apply(redact_text)
)

# 7. Calculate total substitutions
total_substitutions = (
    email_count + phone_count + order_count
)

print("PERSONAL INFORMATION REDACTION RESULTS")
print("=" * 55)

print("Email substitutions:", email_count)
print("Phone substitutions:", phone_count)
print("Order ID substitutions:", order_count)
print("Total substitutions:", total_substitutions)

# 8. Verify no sensitive information remains
remaining_emails = 0
remaining_phones = 0
remaining_orders = 0

for text in redacted_tickets["redacted_text"].dropna():

    remaining_emails += len(
        re.findall(EMAIL_PATTERN, text)
    )

    remaining_phones += len(
        re.findall(PHONE_PATTERN, text)
    )

    remaining_orders += len(
        re.findall(ORDER_PATTERN, text)
    )

print("\nPRIVACY VALIDATION")
print("=" * 55)

print("Remaining emails:", remaining_emails)
print("Remaining phone numbers:", remaining_phones)
print("Remaining order IDs:", remaining_orders)

assert remaining_emails == 0
assert remaining_phones == 0
assert remaining_orders == 0

print("\nSUCCESS: No sensitive information remains!")


# 9. Save only privacy-safe ticket data
safe_columns = [
    "ticket_id",
    "customer_id",
    "true_category",
    "redacted_text"
]

safe_tickets = redacted_tickets[safe_columns].copy()

safe_tickets.to_csv(
    "outputs/redacted_tickets.csv",
    index=False
)

print("\nRedacted tickets saved successfully!")
print("File: outputs/redacted_tickets.csv")
print("Exported columns:", safe_columns)