import streamlit as st
import pandas as pd
import re
import langdetect
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.set_page_config(page_title="Analisis Sentiment Ulasan", page_icon="./images/code.png")



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

# Load dataset
df = pd.read_csv('DataBersihTokopediaSentimenLangdetect.csv')  # Ganti dengan path dataset Anda

# Fungsi Utama Halaman
def main():
    st.markdown("<h1 style='text-align: center; color: #50C878;'>Analisis Tipe Barang Tokopedia</h1>", unsafe_allow_html=True)
        
        # Logo dan Gambar
    inven = './images/invent.png'
    inven1 = './images/invent.png'
    sidebar_logo = inven
    main_body_logo = inven1
    st.logo(sidebar_logo, icon_image=main_body_logo)

    st.sidebar.write("**Tentang Analisis Ini:**")
    st.sidebar.write("Aplikasi ini menganalisis ulasan produk di Tokopedia berdasarkan tipe barang yang telah dipilih.")
    st.sidebar.write("Probabilitas sentimen positif dan negatif dihitung untuk membantu Anda menentukan apakah barang tersebut layak dibeli.")

# Fungsi Preprocessing

    # Ambil daftar tipe barang dari dataset
    # Ambil daftar tipe barang dari dataset
    tipe_barang_options = df['Tipe Barang'].unique().tolist()
    tipe_barang_input = st.selectbox('Pilih Tipe Barang:', tipe_barang_options)

    if st.button('Analysis'):
        if tipe_barang_input:
            # Filter berdasarkan tipe barang dari input pengguna
            filtered_df = df[df['Tipe Barang'] == tipe_barang_input]

            # Hitung jumlah sentimen positif dan negatif
            jumlah_positif = len(filtered_df[filtered_df['Sentimen'] == 'positif'])
            jumlah_negatif = len(filtered_df[filtered_df['Sentimen'] == 'negatif'])

            # Tampilkan hasil analisis
            st.markdown("<h3 style='color: #666699; font-size: 25px;'>Hasil Analisis Tipe Barang:</h3>", unsafe_allow_html=True)
            st.markdown(f"**Jumlah Sentimen untuk produk <strong style='color:#22dd88;'>{tipe_barang_input}</strong>**", unsafe_allow_html=True)
            st.markdown(f"**Positif:** {jumlah_positif}")
            st.markdown(f"**Negatif:** {jumlah_negatif}")

            # Hitung dan tampilkan probabilitas sentimen positif
            total_sentimen = jumlah_positif + jumlah_negatif
            if total_sentimen > 0:
                prob_sentimen_positif = jumlah_positif / total_sentimen
            else:
                prob_sentimen_positif = 0  # Jika tidak ada sentimen
            
            st.write(f"**Probabilitas Sentimen Positif:** {round(prob_sentimen_positif * 100, 2)}%")

            # Rekomendasi berdasarkan probabilitas
            if prob_sentimen_positif == 1:
                st.success("Produk Direkomendasikan")
            elif prob_sentimen_positif >= 0.5:
                st.info("Barang Rekomendasi")
            else:
                st.error("Barang Tidak Direkomendasikan")
        else:
            st.error('Silakan pilih tipe barang untuk analisis.')

if __name__ == '__main__':
    main()
