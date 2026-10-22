
from pathlib import Path
from vinay_inference import ContractAnalyzer


def ask_contract_question(analyzer, contract_text):
    """Accept a user question and return the ContractIQ answer."""

    question = input("\nEnter your question about the contract: ").strip()

    if not question:
        print("Please enter a valid question.")
        return

    result = analyzer.answer_question(contract_text, question)

    print("\n--- ContractIQ Answer ---")
    print("Question:", result["question"])
    print("Answer:", result["answer"] or "No reliable answer found.")
    print("Evidence:", result["evidence"] or "No evidence available.")
    print("Source:", result["source"])
    print("Found:", result["found"])


def main():
    # Load the contract
    contract_path = Path("sample_contract.txt")

    contract_text = contract_path.read_text(encoding="utf-8")

    # Load the model only once
    analyzer = ContractAnalyzer()

    # Ask questions repeatedly until the user exits
    while True:
        ask_contract_question(analyzer, contract_text)

        again = input("\nAsk another question? (y/n): ").strip().lower()

        if again != "y":
            print("Exiting ContractIQ.")
            break


if __name__ == "__main__":
    main()
