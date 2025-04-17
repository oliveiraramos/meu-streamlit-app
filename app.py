import streamlit as st
from docx import Document
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================
# CONFIGURAR NLTK NO STREAMLIT CLOUD
# ============================
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.mkdir(nltk_data_path)
nltk.data.path.append(nltk_data_path)

for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == 'punkt' else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path)

# ============================
# FUNÇÕES AUXILIARES
# ============================
def extract_text(file):
    document = Document(file)
    full_text = [para.text for para in document.paragraphs]
    return "\n".join(full_text)

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('portuguese'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def plot_similarity_matrix(matrix, labels):
    matrix_percent = matrix * 100
    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.matshow(matrix_percent, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="left")
    ax.set_yticklabels(labels)
    threshold = 50
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix_percent[i, j]
            text_color = "white" if value > threshold else "black"
            ax.text(j, i, f"{value:.2f}%", va="center", ha="center", color=text_color)
    ax.set_title("Matriz de Similaridade (%)", pad=20)
    st.pyplot(fig)

# ============================
# INTERFACE STREAMLIT
# ============================
st.title("Comparador de Similaridade de Documentos Word")
st.write("Carregue dois ou mais arquivos DOCX para comparar a similaridade entre os textos.")

uploaded_files = st.file_uploader("Selecione os arquivos", type=["docx"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) > 1:
    texts = []
    file_names = []

    for uploaded_file in uploaded_files:
        file_names.append(uploaded_file.name)
        texts.append(extract_text(uploaded_file))

    st.success("Arquivos carregados e textos extraídos com sucesso!")

    processed_texts = [preprocess_text(text) for text in texts]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    similarity_matrix_percent = similarity_matrix * 100

    df_similarity = pd.DataFrame(similarity_matrix_percent, index=file_names, columns=file_names)
    st.subheader("Matriz de Similaridade (em %)")
    st.dataframe(df_similarity.style.format("{:.2f}%"))

    st.subheader("Heatmap da Matriz de Similaridade")
    plot_similarity_matrix(similarity_matrix, file_names)
else:
    st.info("Por favor, envie pelo menos dois arquivos DOCX para comparação.")
