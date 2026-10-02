from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

MODEL_PATH = r"D:\ContractIntelligence\bert-cuad-qa"

# Load ORIGINAL tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "deepset/bert-base-cased-squad2",
    use_fast=False
)

# Load YOUR trained model
model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_PATH
)

model.eval()


def answer_question(question, context):

    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    offset_mapping = inputs["offset_mapping"]

    best_answer = ""
    best_score = float("-inf")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    for i in range(input_ids.shape[0]):

        start_scores = start_logits[i]
        end_scores = end_logits[i]

        start_idx = torch.argmax(start_scores).item()
        end_idx = torch.argmax(end_scores).item()

        if end_idx < start_idx:
            continue

        score = (
            start_scores[start_idx].item()
            + end_scores[end_idx].item()
        )

        offsets = offset_mapping[i]

        start_char = offsets[start_idx][0].item()
        end_char = offsets[end_idx][1].item()

        if start_char == end_char:
            continue

        answer = context[start_char:end_char]

        if score > best_score:
            best_score = score
            best_answer = answer

    return best_answer, best_score

if __name__ == "__main__":

    context = """
    This Agreement is entered into between ABC Technologies Ltd.
    and XYZ Solutions Inc. The agreement shall commence on January 1,
    2025 and shall remain effective for a period of three years.
    Either party may terminate this agreement by providing 30 days
    written notice to the other party.
    """

    question = "What is the notice period for termination?"

    answer, score = answer_question(question, context)

    print("Question:", question)
    print("Answer:", answer)
    print("Score:", score)

