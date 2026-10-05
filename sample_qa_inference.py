import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = r"D:\\ContractIntelligence\\bert-cuad-download"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model loaded successfully!")
print("Device:", device)


# ============================================================
# TEST QUESTION + CONTEXT
# ============================================================

question = "What is the notice period to terminate?"

context = """
This Agreement shall remain in effect for a period of three years.
Either party may terminate this Agreement upon thirty days written notice
to the other party. The termination shall become effective after the
expiration of the thirty-day notice period.
"""


# ============================================================
# TOKENIZATION
# ============================================================

inputs = tokenizer(
    question,
    context,
    return_tensors="pt",
    truncation="only_second",
    max_length=384,
    return_offsets_mapping=True
)

offset_mapping = inputs.pop("offset_mapping")

# Identify which tokens belong to context
sequence_ids = inputs.sequence_ids(0)

inputs = {
    k: v.to(device)
    for k, v in inputs.items()
}


# ============================================================
# MODEL PREDICTION
# ============================================================

with torch.no_grad():
    outputs = model(**inputs)

start_logits = outputs.start_logits[0]
end_logits = outputs.end_logits[0]


# ============================================================
# FIND BEST VALID ANSWER SPAN
# ============================================================

best_score = -float("inf")
best_start = None
best_end = None

for start in range(len(start_logits)):

    # Only context tokens
    if sequence_ids[start] != 1:
        continue

    for end in range(start, min(start + 30, len(end_logits))):

        if sequence_ids[end] != 1:
            continue

        score = (
            start_logits[start].item()
            + end_logits[end].item()
        )

        if score > best_score:
            best_score = score
            best_start = start
            best_end = end


# ============================================================
# EXTRACT ANSWER
# ============================================================

if best_start is not None:

    answer = tokenizer.decode(
        inputs["input_ids"][0][best_start:best_end + 1],
        skip_special_tokens=True
    )

else:
    answer = ""


# ============================================================
# OUTPUT
# ============================================================

print("\n====================================")
print("QUESTION:")
print(question)

print("\nANSWER:")
print(repr(answer))

print("\nSCORE:")
print(best_score)

print("====================================")