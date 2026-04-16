import torch
import spacy
import spacy.cli
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq

os.environ["SPACY_WARNING_IGNORE"] = "W008"


# ---------------------------
# SAFE SPACY LOADER
# ---------------------------
def load_spacy_model(name):
    try:
        return spacy.load(name)
    except OSError:
        spacy.cli.download(name)
        return spacy.load(name)


# ---------------------------
# LOAD MODEL + VOCAB + NLP
# ---------------------------
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    # Load vocab
    en_vocab = torch.load(os.path.join(MODEL_DIR, "en_vocab.pt"), map_location=device)
    de_vocab = torch.load(os.path.join(MODEL_DIR, "de_vocab.pt"), map_location=device)

    # Load spaCy models safely
    en_nlp = load_spacy_model("en_core_web_sm")
    de_nlp = load_spacy_model("de_core_news_sm")

    # Build model
    encoder = Encoder(len(de_vocab), 256, 512, 2, 0.5)
    decoder = Decoder(len(en_vocab), 256, 512, 2, 0.5)

    model = Seq2Seq(encoder, decoder, device).to(device)

    model_path = os.path.join(MODEL_DIR, "Seq2Seq-model.pt")

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    return model, en_vocab, de_vocab, en_nlp, de_nlp, device


# ---------------------------
# TRANSLATION FUNCTION
# ---------------------------
def translate(sentence, model, en_nlp, de_nlp, en_vocab, de_vocab, device):

    tokens = [token.text.lower() for token in de_nlp.tokenizer(sentence)]
    tokens = ["<sos>"] + tokens + ["<eos>"]

    ids = de_vocab.lookup_indices(tokens)
    tensor = torch.LongTensor(ids).unsqueeze(1).to(device)

    hidden, cell = model.encoder(tensor)

    outputs = [en_vocab["<sos>"]]

    for _ in range(25):
        input_tensor = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(input_tensor, hidden, cell)

        pred_token = output.argmax(1).item()
        outputs.append(pred_token)

        if pred_token == en_vocab["<eos>"]:
            break

    return en_vocab.lookup_tokens(outputs)