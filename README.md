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


def preprocess_qa(question, context, seq_length=256):
    inputs = preprocessor.bert_pack_inputs(
        [tf.constant([question])],
        [tf.constant([context])],
        seq_length=seq_length
    )
    return inputs


def get_answer(question, context):
    inputs = preprocess_qa(question, context)

    outputs = qa_model(inputs)

    start_logits = outputs["start_logits"][0].numpy()
    end_logits = outputs["end_logits"][0].numpy()

    input_word_ids = inputs["input_word_ids"][0].numpy()

    start_index = int(np.argmax(start_logits))
    end_index = int(np.argmax(end_logits))

    if end_index < start_index:
        return "No clear answer found."

    vocab = preprocessor.tokenize.get_vocabulary()

    tokens = []
    for token_id in input_word_ids[start_index:end_index + 1]:
        if token_id == 0:
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
