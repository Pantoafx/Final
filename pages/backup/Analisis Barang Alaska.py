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
    st.logo(sidebar_logo, icon_image=main_body_logo)
    
    col11, col12, col13 = st.columns(3)
    col11.image(toped, use_column_width=True)

    # Sidebar
    
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

            # Menampilkan hasil analisis
            st.markdown("<h3 style='color: #666699; font-size: 25px;'>Hasil Analisis Tipe Barang:</h3>", unsafe_allow_html=True)
            st.markdown(f"**Jumlah Sentimen untuk produk <strong style='color:#22dd88;'>{tipe_barang_input}</strong>**", unsafe_allow_html=True)
            st.markdown(f"<span style='color: green;'>Positif (Model):</span> {jumlah_positif_model}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: red;'>Negatif (Model):</span> {jumlah_negatif_model}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: green;'>Positif (Rating):</span> {jumlah_positif_rating}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: red;'>Negatif (Rating):</span> {jumlah_negatif_rating}", unsafe_allow_html=True)

            # Menghitung probabilitas sentimen positif dari model
            total_sentimen_model = jumlah_positif_model + jumlah_negatif_model
            prob_sentimen_positif_model = jumlah_positif_model / total_sentimen_model if total_sentimen_model > 0 else 0

            st.write(f"**Probabilitas Sentimen Positif (Model):** {round(prob_sentimen_positif_model * 100, 2)}%")

            # Menghitung probabilitas sentimen positif dari rating
            total_sentimen_rating = jumlah_positif_rating + jumlah_negatif_rating
            prob_sentimen_positif_rating = jumlah_positif_rating / total_sentimen_rating if total_sentimen_rating > 0 else 0

            st.write(f"**Probabilitas Sentimen Positif (Rating):** {round(prob_sentimen_positif_rating * 100, 2)}%")

            # Rekomendasi berdasarkan prediksi model
            if prob_sentimen_positif_model >= 0.5:
                st.success("Produk Direkomendasikan berdasarkan prediksi model.")
            else:
                st.error("Produk Tidak Direkomendasikan berdasarkan prediksi model.")

            # Rekomendasi berdasarkan rating
            if prob_sentimen_positif_rating >= 0.5:
                st.success("Produk Direkomendasikan berdasarkan rating.")
            else:
                st.error("Produk Tidak Direkomendasikan berdasarkan rating.")

        else:
            st.error('Silakan pilih tipe barang untuk analisis.')

if __name__ == '__main__':
    main()
