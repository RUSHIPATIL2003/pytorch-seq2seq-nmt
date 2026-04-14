import torch
import spacy

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab
    en_vocab = torch.load("models/en_vocab.pt")
    de_vocab = torch.load("models/de_vocab.pt")

    # Load spacy
    en_nlp = spacy.load("en_core_web_sm")
    de_nlp = spacy.load("de_core_news_sm")

    # Build model
    encoder = Encoder(len(de_vocab), 256, 512, 2, 0.5)
    decoder = Decoder(len(en_vocab), 256, 512, 2, 0.5)

    model = Seq2Seq(encoder, decoder, device).to(device)

    model.load_state_dict(
        torch.load("models/Seq2Seq-model.pt", map_location=device)
    )

    model.eval()

    return model, en_vocab, de_vocab, en_nlp, de_nlp, device


def translate(sentence, model, en_nlp, de_nlp, en_vocab, de_vocab, device):
    tokens = [token.text.lower() for token in de_nlp.tokenizer(sentence)]
    tokens = ["<sos>"] + tokens + ["<eos>"]

    ids = de_vocab.lookup_indices(tokens)
    tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)

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