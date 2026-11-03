
import os
import pandas as pd
import mock_llm

import policy_retrieval as pr
from policy_assistant import answer


POLICY_DIR = "data/policies"
EVAL_FILE = "data/raw/policy_eval.csv"


# Create policy chunks with configurable overlap
def create_chunks(overlap):

    all_chunks = []
    chunk_size = 40
    step_size = chunk_size - overlap

    files = sorted(
        f for f in os.listdir(POLICY_DIR)
        if f.endswith(".txt")
    )

    for filename in files:

        path = os.path.join(POLICY_DIR, filename)

        with open(path, "r", encoding="utf-8") as file:
            words = file.read().split()

        start = 0

        while start < len(words):

            end = min(start + chunk_size, len(words))

            all_chunks.append({
                "source": filename,
                "start_word": start,
                "end_word": end - 1,
                "text": " ".join(words[start:end])
            })

            if end >= len(words):
                break

            start += step_size

    return all_chunks


# Run the evaluation
def evaluate(overlap, label):

    print("\n", "=" * 55)
    print(label)
    print("=" * 55)

    # Build and load the selected chunk index
    chunks = create_chunks(overlap)
    pr.build_index(chunks)

    print("Total chunks:", len(chunks))

    # Reset mock before the full evaluation
    mock_llm.reset()

    eval_df = pd.read_csv(EVAL_FILE)

    retrieval_correct = 0
    fact_correct = 0
    fallback_correct = 0

    for _, row in eval_df.iterrows():

        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        # Retrieve top 3 policy chunks
        retrieved = pr.retrieve(question, k=3)

        retrieved_sources = [
            item["source"] for item in retrieved
        ]

        # Generate grounded answer
        result = answer(question)
        model_answer = result["answer"]

        if expected_doc != "NONE":

            if expected_doc in retrieved_sources:
                retrieval_correct += 1

            if str(expected_fact) in model_answer:
                fact_correct += 1

        else:

            if model_answer.startswith("I could not find"):
                fallback_correct += 1

    print("\nEvaluation Results:")
    print("Answerable questions:", 12)
    print("Expected document retrieved:", retrieval_correct, "/ 12")
    print("Expected fact found:", fact_correct, "/ 12")
    print("Unanswerable fallback:", fallback_correct, "/ 2")

    return {
        "Chunking": label,
        "Total chunks": len(chunks),
        "Document retrieval": f"{retrieval_correct}/12",
        "Expected fact": f"{fact_correct}/12",
        "Fallback": f"{fallback_correct}/2"
    }


if __name__ == "__main__":

    overlap_results = evaluate(
        overlap=10,
        label="40 WORDS WITH 10-WORD OVERLAP"
    )

    zero_results = evaluate(
        overlap=0,
        label="40 WORDS WITH ZERO OVERLAP"
    )

    comparison = pd.DataFrame([
        overlap_results,
        zero_results
    ])

    print("\nFINAL COMPARISON:")
    print(comparison.to_string(index=False))

    os.makedirs("outputs", exist_ok=True)

    comparison.to_csv(
        "outputs/policy_evaluation.csv",
        index=False
    )

    print("\nSaved: outputs/policy_evaluation.csv")
    print("\nTASK 4 EVALUATION COMPLETED!")
