import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np

QA_MODEL_URL = "https://tfhub.dev/see--/bert-uncased-tf2-qa/1"
PREPROCESSOR_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

print("Loading BERT QA model...")
qa_model = hub.load(QA_MODEL_URL)

print("Loading BERT preprocessor...")
preprocessor = hub.load(PREPROCESSOR_URL)

context = """
TensorFlow is an end-to-end open-source platform for machine learning. It has a comprehensive,
flexible ecosystem of tools, libraries and community resources that lets researchers innovate
with machine learning and productionize AI easily.
"""

question = "What is TensorFlow used for?"


def preprocess_qa(question, context):
    question_tensor = tf.constant([question])
    context_tensor = tf.constant([context])

    question_tokens = preprocessor.tokenize(question_tensor)
    context_tokens = preprocessor.tokenize(context_tensor)

    inputs = preprocessor.bert_pack_inputs(
        [question_tokens, context_tokens],
        seq_length=256
    )

    return inputs


def get_answer(question, context):
    inputs = preprocess_qa(question, context)

    model_inputs = [
        inputs["input_word_ids"],
        inputs["input_mask"],
        inputs["input_type_ids"]
    ]

    outputs = qa_model(model_inputs)

    start_logits = outputs[0][0].numpy()
    end_logits = outputs[1][0].numpy()

    input_word_ids = inputs["input_word_ids"][0].numpy()
    input_type_ids = inputs["input_type_ids"][0].numpy()

    context_mask = input_type_ids == 1
    start_logits = np.where(context_mask, start_logits, -1e9)
    end_logits = np.where(context_mask, end_logits, -1e9)

    max_score = -1e9
    start, end = 0, 0

    for i in range(len(start_logits)):
        for j in range(i, min(i + 20, len(end_logits))):
            score = start_logits[i] + end_logits[j]
            if score > max_score:
                max_score = score
                start, end = i, j

    if end < start:
        return "No clear answer found."

    with open("vocab.txt", "r", encoding="utf-8") as f:
        vocab = f.read().splitlines()

    tokens = []

    for token_id in input_word_ids[start:end + 1]:
        token_id = int(token_id)

        if token_id < 0 or token_id >= len(vocab):
            continue

        token = vocab[token_id]

        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        tokens.append(token)

    answer = " ".join(tokens)
    answer = answer.replace(" ##", "")

    return answer.strip()


answer = get_answer(question, context)

print("\nQuestion:", question)
print("Answer:", answer)
