import sys
import os
import torch
import spacy

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from app.inference import load_model, translate


def main():
    print("🔄 Loading model...")
    model, en_vocab, de_vocab, en_nlp, de_nlp, device = load_model()

    print("✅ Model loaded successfully!\n")

    while True:
        sentence = input("Enter German sentence (or 'q' to quit): ")

        if sentence.lower() == "q":
            break

        if not sentence.strip():
            print("⚠️ Empty input, try again.\n")
            continue

        try:
            output = translate(
                sentence,
                model,
                en_nlp,
                de_nlp,
                en_vocab,
                de_vocab,
                device
            )

            cleaned = [tok for tok in output if tok not in ["<sos>", "<eos>"]]

            print("👉 Translation:", " ".join(cleaned), "\n")

        except Exception as e:
            print("❌ Error during translation:")
            print(e, "\n")


if __name__ == "__main__":
    main()