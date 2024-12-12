import streamlit as st
import pandas as pd
import re
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Alaska", page_icon="./images/code.png")

# Fungsi preprocessing teks
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)  # Hanya menyisakan alphanumeric dan spasi
    text = text.lower()  # Mengubah ke huruf kecil
    stemmer = StemmerFactory().create_stemmer()
    text = stemmer.stem(text)  # Stemming kata
    return text

# Fungsi prediksi sentimen menggunakan model yang dimuat dari file .pkl
def predict_sentiment(text, model):
    # Preprocessing teks
    processed_text = preprocess_text(text)
    # Prediksi menggunakan model SVM
    result = model.predict([processed_text])
    return result[0]

# Load model SVM dari file .pkl
with open('SVM_Pipe.pkl', 'rb') as model_file:
    modelsvc_loaded = pickle.load(model_file)

# Load dataset
df = pd.read_csv('DataAlaska.csv')

# Fungsi utama halaman
def main():
    # Menampilkan logo dan gambar
    alas = './images/alaska.png'
    toped = './images/logotoped.png'
    sidebar_logo = alas
    main_body_logo = alas
    # Sidebar
    st.logo(sidebar_logo, icon_image=main_body_logo)

    # Header halaman utama
    st.markdown("<h1 style='text-align: center; color: #50C878;'>Analisis Tipe Barang & Ulasan Alaska</h1>", unsafe_allow_html=True)

    # Memproses tipe barang
    tipe_barang_options = df['Tipe Barang'].unique().tolist()
    tipe_barang_input = st.selectbox('Pilih Tipe Barang:', tipe_barang_options)

    if st.button('Analisis'):
        if tipe_barang_input:
            # Filter berdasarkan tipe barang
            filtered_df = df[df['Tipe Barang'] == tipe_barang_input]

            # Menghitung prediksi sentimen menggunakan model SVM yang dimuat dari file pkl
            filtered_df['Prediksi Sentimen'] = filtered_df['Ulasan'].apply(lambda x: predict_sentiment(x, modelsvc_loaded))

            # Menghitung jumlah sentimen positif dan negatif dari prediksi model
            jumlah_positif_model = len(filtered_df[filtered_df['Prediksi Sentimen'] == 'positif'])
            jumlah_negatif_model = len(filtered_df[filtered_df['Prediksi Sentimen'] == 'negatif'])

            # Menghitung jumlah ulasan positif dan negatif berdasarkan rating
            jumlah_positif_rating = len(filtered_df[filtered_df['Rating'] >= 3])
            jumlah_negatif_rating = len(filtered_df[filtered_df['Rating'] < 3])

            # Menampilkan hasil analisis dalam dua kolom
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<h3 style='color: #666699; font-size: 25px;'>Hasil Berdasarkan Rating:</h3>", unsafe_allow_html=True)
                st.write(f"Rating ★ (3-5): {jumlah_positif_rating}")
                st.write(f"Rating ★ (1-2): {jumlah_negatif_rating}")

            with col2:
                st.markdown("<h3 style='color: #666699; font-size: 25px;'>Hasil Analisis SVM:</h3>", unsafe_allow_html=True)
                st.write(f"Positif ✔️ (SVM): {jumlah_positif_model}")
                st.write(f"Negatif ❌ (SVM): {jumlah_negatif_model}")

    else:
        st.info('Silakan pilih tipe barang untuk analisis.')

if __name__ == '__main__':
    main()
