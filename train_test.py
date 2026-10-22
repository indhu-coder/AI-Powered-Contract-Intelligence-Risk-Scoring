
import json
import os

files = {
    "train": r"D:\\ContractIntelligence\\cuad-main\\data\\train_separate_questions.json",
    "test": r"D:\\ContractIntelligence\\cuad-main\\data\\test.json",
}

for name, path in files.items():
    print(f"\n--- {name.upper()} ---")

    if not os.path.exists(path):
        print("File not found:", path)
        continue

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Top-level keys:", list(data.keys()))

    if "data" in data:
        print("Documents:", len(data["data"]))

        questions = [
            qa
            for doc in data["data"]
            for paragraph in doc.get("paragraphs", [])
            for qa in paragraph.get("qas", [])
        ]

        print("Questions:", len(questions))
        print(
            "Answerable:",
            sum(not qa.get("is_impossible", False) for qa in questions)
        )
        print(
            "Unanswerable:",
            sum(qa.get("is_impossible", False) for qa in questions)
        )
    else:
        print("Sample structure:", str(data)[:500])


import json

base = r"D:\\ContractIntelligence\\cuad-main\\data"

with open(base + r"\\train_separate_questions.json", encoding="utf-8") as f:
    train_json = json.load(f)

with open(base + r"\\test.json", encoding="utf-8") as f:
    test_json = json.load(f)

train_titles = {
    doc.get("title", "")
    for doc in train_json["data"]
}

test_titles = {
    doc.get("title", "")
    for doc in test_json["data"]
}

overlap = train_titles & test_titles

print("Train documents:", len(train_titles))
print("Test documents:", len(test_titles))
print("Overlapping titles:", len(overlap))
print("Examples:", list(overlap)[:10])



print("Number of documents:", len(train_json["data"]))

total_questions = sum(
    len(paragraph["qas"])
    for document in train_json["data"]
    for paragraph in document["paragraphs"]
)

print("Total questions in JSON:", total_questions)

total_answerable = sum(
    1
    for document in train_json["data"]
    for paragraph in document["paragraphs"]
    for qa in paragraph["qas"]
    if not qa.get("is_impossible", False)
)

total_unanswerable = total_questions - total_answerable

print("Answerable questions in JSON:", total_answerable)
print("Unanswerable questions in JSON:", total_unanswerable)


print("Number of documents:", len(test_json["data"]))

total_questions = sum(
    len(paragraph["qas"])
    for document in test_json["data"]
    for paragraph in document["paragraphs"]
)

print("Total questions in JSON:", total_questions)

total_answerable = sum(
    1
    for document in test_json["data"]
    for paragraph in document["paragraphs"]
    for qa in paragraph["qas"]
    if not qa.get("is_impossible", False)
)

total_unanswerable = total_questions - total_answerable

print("Answerable questions in JSON:", total_answerable)
print("Unanswerable questions in JSON:", total_unanswerable)