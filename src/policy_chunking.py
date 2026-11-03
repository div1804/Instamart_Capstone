
import os
import json

POLICY_DIR = "data/policies"
OUTPUT_DIR = "outputs"

CHUNK_SIZE = 40
OVERLAP = 10
STEP_SIZE = CHUNK_SIZE - OVERLAP


def chunk_document(file_path):

    filename = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    words = text.split()

    chunks = []
    start = 0

    while start < len(words):

        end = min(start + CHUNK_SIZE, len(words))

        chunk_text = " ".join(words[start:end])

        chunks.append({
            "source": filename,
            "start_word": start,
            "end_word": end - 1,
            "text": chunk_text
        })

        # Stop after reaching the document's last word
        if end >= len(words):
            break

        start += STEP_SIZE

    return chunks


def chunk_all_documents():

    all_chunks = []

    policy_files = sorted(
        filename
        for filename in os.listdir(POLICY_DIR)
        if filename.endswith(".txt")
    )

    for filename in policy_files:

        file_path = os.path.join(POLICY_DIR, filename)

        document_chunks = chunk_document(file_path)

        all_chunks.extend(document_chunks)

        print(
            f"{filename}: {len(document_chunks)} chunks"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(
        OUTPUT_DIR,
        "policy_chunks.json"
    )

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nTOTAL CHUNKS:", len(all_chunks))
    print("Saved:", output_file)

    return all_chunks


if __name__ == "__main__":

    chunks = chunk_all_documents()

    print("\nFIRST CHUNK:")
    print(chunks[0])

    print("\nTASK 1 COMPLETED!")
