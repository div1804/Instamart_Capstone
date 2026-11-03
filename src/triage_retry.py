import time
import random
import mock_llm

# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 0.05


def call_with_retry(ticket_id, messages):

    for retry_number in range(MAX_RETRIES + 1):

        try:
            # Call mock LLM
            response = mock_llm.chat(messages)

            return response

        except mock_llm.TransientLLMError:

            # Stop after maximum retries
            if retry_number == MAX_RETRIES:
                print(
                    f"Ticket {ticket_id}: "
                    "Maximum retries reached!"
                )
                raise

            # Calculate exponential backoff with jitter
            wait_time = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            # Log every retry
            print(
                f"Ticket {ticket_id} | "
                f"Retry number: {retry_number} | "
                f"Wait: {wait_time:.4f} seconds"
            )

            time.sleep(wait_time)


print("RETRY FUNCTION CREATED SUCCESSFULLY!")