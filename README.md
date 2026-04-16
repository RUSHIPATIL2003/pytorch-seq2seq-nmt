# 🌍 Sequence-to-Sequence German → English Translator

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![UI](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![Repo Size](https://img.shields.io/github.com/RUSHIPATIL2003/pytorch-seq2seq-nmt)
![Last Commit](https://img.shields.io/github.com/RUSHIPATIL2003/pytorch-seq2seq-nmt)

---

A deep learning–based **German → English translator** built using **PyTorch** and deployed with **Streamlit**.  
This project uses a **Sequence-to-Sequence (Seq2Seq) LSTM Encoder–Decoder architecture** trained on the **Multi30k dataset**.

---

## 🚀 Features

- 🔤 German → English translation
- 🧠 LSTM-based Encoder–Decoder model
- 📦 Custom vocabulary handling
- ⚡ Real-time inference with Streamlit
- 📊 BLEU score evaluation
- 💾 Pretrained model loading (`.pt`)

---

## 🧱 Project Structure

```bash
seq2seq-translator/
│
├── app/
│   ├── app.py              # Streamlit UI
│   └── inference.py        # Model loading + translation logic
│
├── src/
│   └── models/
│       ├── encoder.py
│       ├── decoder.py
│       └── seq2seq.py
│
├── models/
│   ├── Seq2Seq-model.pt    # Trained model weights
│   ├── en_vocab.pt         # English vocabulary
│   └── de_vocab.pt         # German vocabulary
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/RUSHIPATIL2003/pytorch-seq2seq-nmt.git
cd seq2seq-translator
```

---

### 2. Create virtual environment (recommended)

```bash
python -m venv project_env
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Download spaCy models

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

---

## ▶️ Run the App

```bash
streamlit run app/app.py
```

Then open the local URL shown in your terminal.

---

## 🧠 Model Details

* Architecture: Sequence-to-sequence model
* Encoder: Long short-term memory
* Decoder: LSTM
* Embedding Size: 256
* Hidden Size: 512
* Layers: 2
* Dropout: 0.5

---

## 💾 Model & Vocabulary

Make sure the following files exist inside `/models`:

* `Seq2Seq-model.pt`
* `en_vocab.pt`
* `de_vocab.pt`

These are generated after training and are required for inference.

---

## 🏋️ Training (Optional)

Training was originally done in a notebook using:

* Dataset: Multi30k (via Hugging Face)
* Tokenization: spaCy
* Evaluation: BLEU score

To retrain:

1. Use the notebook in `/notebooks`
2. Save artifacts:

```python
torch.save(model.state_dict(), "models/Seq2Seq-model.pt")
torch.save(en_vocab, "models/en_vocab.pt")
torch.save(de_vocab, "models/de_vocab.pt")
```

---

## 🌐 Deployment

### Streamlit Cloud

1. Push this repo to GitHub
2. Go to Streamlit Cloud
3. Select your repository
4. Set entry point:

```bash
app/app.py
```

---

## 📚 References

This project is based on foundational work in **Neural Machine Translation and Seq2Seq models**:

### 🔬 Research Papers

1. **Sequence to Sequence Learning with Neural Networks**
   Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2014)
   🔗 https://arxiv.org/abs/1409.3215

2. **Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation**
   Kyunghyun Cho et al. (2014)
   🔗 https://arxiv.org/abs/1406.1078

3. **Neural Machine Translation by Jointly Learning to Align and Translate**
   Dzmitry Bahdanau et al. (2014)
   🔗 https://arxiv.org/abs/1409.0473


---

### 📘 Key Concepts & Tools

* Sequence-to-sequence model
* Long short-term memory
* BLEU score
* spaCy
* PyTorch

---

## 🔥 Future Improvements

* ✨ Add Attention mechanism
* 🔄 Replace LSTM with Transformer
* 🌍 Multi-language support
* 📱 Improved UI/UX
* ⚡ FastAPI backend

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Rushi Patil**
GitHub: https://github.com/RUSHIPATIL2003

---
