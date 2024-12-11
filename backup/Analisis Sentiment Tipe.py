import streamlit as st
import pandas as pd
import pickle
import re
import langdetect
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import time

# Fungsi Preprocessing
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()
    stemmer = StemmerFactory().create_stemmer()
    text = stemmer.stem(text)
    return text

# Fungsi Deteksi Bahasa
def detect_language(text):
    try:
        lang = langdetect.detect(text)
        return lang
    except:
        return 'unknown'

# Load model dari file .pkl
with open('SVM_Pipe.pkl', 'rb') as model_file:
    modelsvc_loaded = pickle.load(model_file)

with open('SVM_Pipe_Tipe.pkl', 'rb') as model_barang_file:
    model_barang_loaded = pickle.load(model_barang_file)

# Fungsi Utama Halaman
def main():
    st.markdown("<h1 style='text-align: center; color: #50C878;'>Analisis Sentimen dan Tipe Barang</h1>", unsafe_allow_html=True)
    
    userText = st.text_input('Masukkan Ulasan Produk:', placeholder='Paste Ulasan Disini..')

    if st.button('Analysis'):
        if userText:
            if len(userText.split()) <= 1:
                st.toast('Mohon masukkan lebih dari satu kata untuk analisis sentimen.')
            else:
                st.info(f"*Ulasan yang diinput:*\n\n{userText}")

                lang = detect_language(userText)
                
                if lang == 'id':
                    text_clean = preprocess_text(userText)
                    bar = st.progress(0)
                    time.sleep(0.5)
                    bar.progress(50)
                    time.sleep(1)
                    bar.progress(100)

                    st.success(f"*Ulasan yang sudah diproses:*\n\n{text_clean}")

                    # Prediksi Sentimen
                    text_vector = modelsvc_loaded['vectorizer'].transform([text_clean])
                    prediction_proba = modelsvc_loaded['classifier'].predict_proba(text_vector)
                    proba_positif = prediction_proba[0][1]
                    rating = round(proba_positif * 5, 2)
                    sentiment_label = 'positif' if proba_positif >= 0.5 else 'negatif'

                    # Prediksi Tipe Barang
                    text_vector_barang = model_barang_loaded['vectorizer'].transform([text_clean])
                    pred_tipe_barang = model_barang_loaded['classifier'].predict(text_vector_barang)

                    # Tampilkan hasil analisis
                    st.markdown("<h3 style='color: #666699; font-size: 25px;'>Hasil Analisis:</h3>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)

                    if sentiment_label == 'positif':
                        col1.image('./images/mots.gif', use_column_width=True, width=100)
                    else:
                        col1.image('./images/motn.gif', use_column_width=True, width=100)

                    col3.metric("Perkiraan Rating", rating, None)
                    col3.markdown(f"<h4>Tipe Barang: {pred_tipe_barang[0]}</h4>", unsafe_allow_html=True)
                    if sentiment_label == 'positif':
                        col2.markdown("<h2 style='color: #50C878;'> Positif </h2>", unsafe_allow_html=True)
                    else:
                        col2.markdown("<h2 style='color: #FF4B4B;'> Negatif </h2>", unsafe_allow_html=True)

                else:
                    st.warning('Mohon masukkan teks dalam Bahasa Indonesia.')
        else:
            st.error('Masukkan teks untuk analisis.')

if __name__ == '__main__':
    main()
