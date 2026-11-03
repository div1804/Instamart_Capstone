import mock_llm
from triage_retry import call_with_retry
from policy_retrieval import retrieve


SYSTEM_PROMPT = """
You are a POLICY ASSISTANT.

Answer customer questions using only the supplied policy context.

Do not use outside knowledge or make up information.

If the answer cannot be found in the supplied context,
respond with:
I could not find the information in the supplied policies.
"""


def answer(question):

    # Retrieve the top 3 relevant policy chunks
    results = retrieve(question, k=3)

    # Build the exact required user message
    context_parts = []

    for result in results:
        context_parts.append(
            f"[{result['source']}]\n{result['text']}"
        )

    user_message = (
        "CONTEXT:\n"
        + "\n\n".join(context_parts)
        + "\n\nQUESTION:\n"
        + question
    )

    # Send exactly two messages to mock LLM
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    response = call_with_retry("POLICY_EVAL", messages)

    model_answer = response.choices[0].message.content

    sources = list(dict.fromkeys(
        result["source"] for result in results
    ))

    return {
        "answer": model_answer,
        "sources": sources
    }


if __name__ == "__main__":

    question = "What is the refund policy for cancelled orders?"

    result = answer(question)

    print("QUESTION:")
    print(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print("-", source)
