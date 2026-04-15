import streamlit as st
from inference import load_model, translate
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Translator")

st.title("🌍 German → English Translator")

@st.cache_resource
def init():
    return load_model()

model, en_vocab, de_vocab, en_nlp, de_nlp, device = init()

sentence = st.text_input("Enter German sentence")

if st.button("Translate"):
    if sentence.strip():
        output = translate(
            sentence,
            model,
            en_nlp,
            de_nlp,
            en_vocab,
            de_vocab,
            device
        )
        st.success(" ".join(output[1:-1]))