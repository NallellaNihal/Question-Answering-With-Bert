# 🤖 BERT Question Answering System

An end-to-end **Extractive Question Answering system** built using a pretrained BERT model from TensorFlow Hub.
This project demonstrates how deep learning models can extract precise answers from unstructured text.

---

## 🚀 Overview

This system takes:

* 📄 A **context paragraph**
* ❓ A **natural language question**

and returns:

* ✅ The **most relevant answer span** from the context using BERT

---

## 🧠 Key Features

* 🔍 Extractive QA using **BERT (Bidirectional Encoder Representations from Transformers)**
* ⚡ TensorFlow + TensorFlow Hub integration
* 🧩 Tokenization & input preprocessing with BERT pipeline
* 📊 Custom span selection logic for accurate answer extraction
* 🧼 Clean and modular Python implementation

---

## 🛠 Tech Stack

* **Python**
* **TensorFlow**
* **TensorFlow Hub**
* **TensorFlow Text**
* **NumPy**
* **BERT (NLP Model)**

---

## 📂 Project Structure

```bash
bert-qa/
│── main.py            # Core QA pipeline
│── vocab.txt          # BERT vocabulary file
│── requirements.txt   # Dependencies
│── README.md          # Documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/NallellaNihal/Question-Answering-With-Bert.git
cd Question-Answering-With-Bert
```

### 2. Create virtual environment

```bash
python3.11 -m venv bert-env
source bert-env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the application:

```bash
python main.py
```

---

## 📌 Example

**Context:**

> TensorFlow is an end-to-end open-source platform for machine learning...

**Question:**

> What is TensorFlow used for?

**Output:**

```bash
Answer: machine learning
```

---

## 🧩 How It Works

1. Input question & context
2. Tokenization using BERT preprocessor
3. Input encoding (`input_ids`, `mask`, `type_ids`)
4. BERT QA model predicts:

   * Start logits
   * End logits
5. Custom span scoring selects best answer
6. Tokens mapped back to readable text using vocabulary

---

## 🔥 Applications

* 💬 Chatbots & Virtual Assistants
* 📄 Document Question Answering
* 📚 Knowledge Retrieval Systems
* ⚖️ Legal & Contract Analysis
* 🏥 Medical Information Extraction

---

## 🚧 Future Improvements

* 🌐 Streamlit-based web UI
* 📚 Multi-document question answering
* 🔍 Semantic search integration
* ⚡ Performance optimization
* ☁️ Deployment (Docker / Cloud)

---

## 👨‍💻 Author

**Nihal Nallella**
🔗 GitHub: https://github.com/NallellaNihal
🔗 LinkedIn: https://linkedin.com/in/nallella-nihal

---

## ⭐ Contribute

Feel free to fork, improve, and submit pull requests!

---

## 📜 License

This project is open-source and available under the MIT License.

---
