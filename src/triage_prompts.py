# ==========================================
# ZERO-SHOT SYSTEM PROMPT
# ==========================================

ZERO_SHOT_PROMPT = """
You are a support ticket classification assistant.

CLASSIFY the customer support ticket into exactly one
of the following allowed categories:

1. billing
2. delivery
3. technical
4. fraud

Read the ticket carefully and identify its main issue.

Return a JSON-only response in this exact format:
{"category": "...", "reason": "..."}

Do not include markdown, explanations outside JSON,
or any additional text.
"""


# ==========================================
# FEW-SHOT SYSTEM PROMPT
# ==========================================

FEW_SHOT_PROMPT = """
You are an expert support ticket classification assistant.

CLASSIFY the customer support ticket into exactly one
of the following allowed categories:

1. billing
2. delivery
3. technical
4. fraud

EXAMPLES:

Ticket: I was charged twice for my subscription.
Response: {"category": "billing", "reason": "Duplicate payment"}

Ticket: My order has not arrived yet.
Response: {"category": "delivery", "reason": "Order delivery delay"}

Ticket: Someone made an unauthorized transaction using my account.
Response: {"category": "fraud", "reason": "Unauthorized transaction"}

Ticket: The mobile application keeps crashing.
Response: {"category": "technical", "reason": "Application malfunction"}

Now classify the customer's ticket.

Return a JSON-only response in this exact format:
{"category": "...", "reason": "..."}

Do not include markdown, explanations outside JSON,
or any additional text.
"""
# ==========================================
# PROMPT VALIDATION
# ==========================================

assert "CLASSIFY" in ZERO_SHOT_PROMPT
assert "CLASSIFY" in FEW_SHOT_PROMPT

assert "EXAMPLES:" not in ZERO_SHOT_PROMPT
assert "EXAMPLES:" in FEW_SHOT_PROMPT

allowed_categories = [
    "billing",
    "delivery",
    "technical",
    "fraud"
]

for category in allowed_categories:
    assert category in ZERO_SHOT_PROMPT
    assert category in FEW_SHOT_PROMPT

assert '{"category": "...", "reason": "..."}' in ZERO_SHOT_PROMPT
assert '{"category": "...", "reason": "..."}' in FEW_SHOT_PROMPT

print("ZERO-SHOT PROMPT VALIDATED!")
print("FEW-SHOT PROMPT VALIDATED!")
print("All four categories are present!")
print("JSON response format validated!")
print("\nPROMPT CREATION COMPLETED!")