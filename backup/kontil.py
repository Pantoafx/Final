import streamlit as st
import pickle
import re
import langdetect
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import time

# Set favicon
st.set_page_config(page_title="Analisis Sentiment Ulasan", page_icon="./images/code.png")

# Kamus Normalisasi
norm = {
    'knp': 'kenapa', 'bgs': 'bagus', 'syg': 'sayang', 'bs': 'bisa', 'w': 'saya',
    # ... (tambahkan sisa kamus normalisasi Anda)
}

# Fungsi Normalisasi
def normalisasi(text):
    for key, value in norm.items():
        pattern = r'\b{}\b'.format(re.escape(key))
        text = re.sub(pattern, value, text)
    return text

# Load model dari file .pkl
with open('SVM_Pipe.pkl', 'rb') as model_file:
    modelsvc_loaded = pickle.load(model_file)

with open('SVM_Pipe_Tipe.pkl', 'rb') as model_file_barang:
    model_barang_loaded = pickle.load(model_file_barang)

# Fungsi Preprocessing
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()
    text = normalisasi(text)
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

# Fungsi Utama Halaman
def main(): 
    st.title("Analisis Sentimen Ulasan Tokopedia")
    
    # Input Teks dari Pengguna
    userText = st.text_input('Masukkan Ulasan Produk:', placeholder='Paste Ulasan Disini..')
    
    # Tombol Analisis Sentimen
    if st.button('Analisis'):
        if userText:
            if len(userText.split()) <= 1:
                st.warning('Mohon masukkan lebih dari satu kata untuk analisis sentimen.')
            else:
                # Tampilkan Ulasan Asli
                st.info(f"*Ulasan yang diinput:*\n\n{userText}")

                # Deteksi Bahasa
                lang = detect_language(userText)
                
                if lang == 'id':
                    text_clean = preprocess_text(userText)
                    bar = st.progress(0)
                    time.sleep(0.5)
                    bar.progress(50)
                    time.sleep(1)
                    bar.progress(100)
        
                    st.success(f"*Ulasan yang sudah diproses:*\n\n{text_clean}")
                    
                    # Transformasi Teks dan Prediksi Sentimen
                    text_vector = modelsvc_loaded['vectorizer'].transform([text_clean])
                    prediction_proba = modelsvc_loaded['classifier'].predict_proba(text_vector)
                    
                    proba_positif = prediction_proba[0][1]
                    rating = predict_rating(proba_positif)
                    sentiment_label = 'positif' if proba_positif >= 0.5 else 'negatif'
                    
                    # Prediksi Tipe Barang
                    prediction_tipe_barang = model_barang_loaded.predict([text_clean])[0]

                    # Tentukan Layak atau Tidak
                    status_barang = "Layak Dibeli" if sentiment_label == 'positif' else "Barang ini Tidak Layak Dibeli"

                    # Tampilkan Hasil
                    st.markdown("<h3 style='color: #666699;'>Hasil Analisis Sentimen:</h3>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)

                    # Tampilkan GIF
                    if sentiment_label == 'positif':
                        col1.image('./images/mots.gif', use_column_width=True, width=100)
                    else:
                        col1.image('./images/motn.gif', use_column_width=True, width=100)

                    col3.metric("Perkiraan Rating", rating, None)
                    col3.metric("Probabilitas Sentimen SVM", f"{round(proba_positif * 100, 2)}%")

                    # Tampilkan Label Sentimen
                    if sentiment_label == 'positif':
                        col2.markdown("<h2 style='color: #50C878;'>Positif</h2>", unsafe_allow_html=True)
                    else:
                        col2.markdown("<h2 style='color: #FF4B4B;'>Negatif</h2>", unsafe_allow_html=True)

                    # Tampilkan Tipe Barang dan Status
                    st.markdown(f"**Tipe Barang:** {prediction_tipe_barang}")
                    st.markdown(f"**Status:** {status_barang}")

if __name__ == '__main__':
    main()
