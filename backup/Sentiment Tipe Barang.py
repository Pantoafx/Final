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

# Fungsi Prediksi Rating
def predict_rating(proba):
    return round(proba * 5, 2)

# Load model dari file .pkl
with open('SVM_Pipe.pkl', 'rb') as model_file:
    modelsvc_loaded = pickle.load(model_file)

# Load model dari file .pkl untuk tipe barang
with open('SVM_PipeBarang.pkl', 'rb') as model_barang_file:
    model_barang_loaded = pickle.load(model_barang_file)

# Load dataset
df = pd.read_csv('DataBersihTokopediaSentimenLangdetect.csv')  # Ganti dengan path dataset Anda

# Fungsi Utama Halaman
def main(): 
    st.markdown("<h1 style='text-align: center; color: #50C878;'>Analisis Sentimen Ulasan Tokopedia</h1>", unsafe_allow_html=True)
    
    userText = st.text_input('Masukkan Ulasan Produk:', placeholder='Paste Ulasan Disini..')

    # Ambil daftar tipe barang dari dataset
    tipe_barang_options = df['Tipe Barang'].unique().tolist()
    tipe_barang_input = st.selectbox('Pilih Tipe Barang:', tipe_barang_options)

    if st.button('Analysis'):
        if userText and tipe_barang_input:
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
                    
                    # Prediksi Tipe Barang
                    text_vector_barang = model_barang_loaded['vectorizer'].transform([text_clean])
                    pred_tipe_barang = model_barang_loaded['classifier'].predict(text_vector_barang)

                    # Prediksi Sentimen
                    text_vector = modelsvc_loaded['vectorizer'].transform([text_clean])
                    prediction_proba = modelsvc_loaded['classifier'].predict_proba(text_vector)
                    
                    proba_positif = prediction_proba[0][1]
                    rating = predict_rating(proba_positif)
                    sentiment_label = 'positif' if proba_positif >= 0.5 else 'negatif'

                    # Filter berdasarkan tipe barang dari input pengguna
                    filtered_df = df[df['Tipe Barang'] == tipe_barang_input]

                    # Hitung jumlah sentimen positif dan negatif
                    jumlah_positif = len(filtered_df[filtered_df['Sentimen'] == 'positif'])
                    jumlah_negatif = len(filtered_df[filtered_df['Sentimen'] == 'negatif'])

                    # Tampilkan hasil analisis
                    st.markdown("<h3 style='color: #666699; font-family: Source Sans Pro, sans-serif; font-size: 25px; margin-bottom: 20px; margin-top: 50px;'>Hasil Analisis Sentimen:</h3>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)

                    if sentiment_label == 'positif':
                        col1.image('./images/mots.gif', use_column_width=True, width=100)
                    else:
                        col1.image('./images/motn.gif', use_column_width=True, width=100)

                    col3.metric("Perkiraan Rating", rating, None)
                    col3.markdown(f"<h4>Tipe Barang: {tipe_barang_input}</h4>", unsafe_allow_html=True)
                    if sentiment_label == 'positif':
                        col2.markdown("<h2 style='color: #50C878;'> Positif </h2>", unsafe_allow_html=True)
                    else:
                        col2.markdown("<h2 style='color: #FF4B4B;'> Negatif </h2>", unsafe_allow_html=True)

                    # Tampilkan jumlah sentimen dari produk yang sama
                    st.markdown(f"**Jumlah Sentimen untuk produk <strong style='color:#22dd88;'>{tipe_barang_input}</strong>**", unsafe_allow_html=True)
                    st.markdown(f"**Positif:** {jumlah_positif}")
                    st.markdown(f"**Negatif:** {jumlah_negatif}")

                    # Tampilkan status apakah barang layak dibeli atau tidak
                    if jumlah_positif > jumlah_negatif:
                        st.success("Status Barang: Layak Dibeli")
                    else:
                        st.error("Status Barang: Tidak Layak Dibeli")
                
                else:
                    st.warning('Mohon masukkan teks dalam Bahasa Indonesia.')
        else:
            st.error('Masukkan teks dan tipe barang untuk analisis.')

if __name__ == '__main__':
    main()
