import json
import re

ALLOWED_CATEGORIES = {
    "billing",
    "delivery",
    "technical",
    "fraud"
}


def validate_response(response_text):

    # Step 1: Try parsing JSON
    try:
        data = json.loads(response_text)

        # Valid JSON must have an allowed category
        if not isinstance(data, dict):
            return {
                "status": "flagged_invalid_output",
                "category": "",
                "reason": "Human review required."
            }

        category = data.get("category")

        if isinstance(category, str):
            category = category.lower().strip()

            if category in ALLOWED_CATEGORIES:
                return {
                    "status": "classified_json",
                    "category": category,
                    "reason": data.get("reason", "")
                }

        # Valid JSON with an invalid category
        return {
            "status": "flagged_invalid_output",
            "category": "",
            "reason": "Invalid JSON category. Human review required."
        }

    except (json.JSONDecodeError, TypeError):
        pass

    # Step 2: Only for non-JSON responses
    words = re.findall(r"[a-z]+", response_text.lower())

    matched_categories = [
        word for word in words
        if word in ALLOWED_CATEGORIES
    ]

    if len(matched_categories) == 1:
        return {
            "status": "classified_keyword",
            "category": matched_categories[0],
            "reason": ""
        }

    # Step 3: Human review
    return {
        "status": "flagged_invalid_output",
        "category": "",
        "reason": "Invalid model response. Human review required."
    }


print("RESPONSE VALIDATION FUNCTION CREATED SUCCESSFULLY!")