import streamlit as st
import pandas as pd
import re
import langdetect
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Sentimen dan Tipe Barang Tokopedia", page_icon="./images/code.png")

# Fungsi preprocessing teks
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)  # Hanya menyisakan alphanumeric dan spasi
    text = text.lower()  # Mengubah ke huruf kecil
    stemmer = StemmerFactory().create_stemmer()
    text = stemmer.stem(text)  # Stemming kata
    return text

# Fungsi deteksi bahasa
def detect_language(text):
    try:
        lang = langdetect.detect(text)
        return lang
    except langdetect.lang_detect_exception.LangDetectException:
        return 'unknown'

# Load dataset
df = pd.read_csv('DataBersihTokopediaSentimenLangdetect.csv')

# Fungsi utama halaman
def main():
    st.markdown("<h1 style='text-align: center; color: #50C878;'>Analisis Tipe Barang Tokopedia</h1>", unsafe_allow_html=True)

    # Menampilkan logo dan gambar
    inven = './images/invent.png'
    gif_positive = './images/mots.gif'
    gif_negative = './images/motn.gif'
    
    # Sidebar
    st.sidebar.image(inven, use_column_width=True)

    # Deskripsi sidebar
    st.sidebar.write("**Tentang Analisis Ini:**")
    st.sidebar.write("Aplikasi ini menganalisis ulasan produk di Tokopedia berdasarkan tipe barang yang dipilih.")
    st.sidebar.write("Probabilitas sentimen positif dan negatif dihitung untuk menentukan apakah produk layak direkomendasikan.")

    # Memproses tipe barang
    tipe_barang_options = df['Tipe Barang'].unique().tolist()
    tipe_barang_input = st.selectbox('Pilih Tipe Barang:', tipe_barang_options)

    if st.button('Analisis'):
        if tipe_barang_input:
            # Filter berdasarkan tipe barang
            filtered_df = df[df['Tipe Barang'] == tipe_barang_input]

            # Cek apakah ada data
            if not filtered_df.empty:
                # Menghitung jumlah sentimen positif dan negatif
                jumlah_positif = len(filtered_df[filtered_df['Sentimen'] == 'positif'])
                jumlah_negatif = len(filtered_df[filtered_df['Sentimen'] == 'negatif'])

                # Menampilkan hasil analisis
                st.markdown("<h3 style='color: #666699; font-size: 25px;'>Hasil Analisis Tipe Barang:</h3>", unsafe_allow_html=True)
                st.markdown(f"**Jumlah Sentimen untuk produk <strong style='color:#22dd88;'>{tipe_barang_input}</strong>**", unsafe_allow_html=True)
                st.markdown(f"<span style='color: green;'>Positif:</span> {jumlah_positif}", unsafe_allow_html=True)
                st.markdown(f"<span style='color: red;'>Negatif:</span> {jumlah_negatif}", unsafe_allow_html=True)

                # Menghitung probabilitas sentimen positif
                total_sentimen = jumlah_positif + jumlah_negatif
                if total_sentimen > 0:
                    prob_sentimen_positif = jumlah_positif / total_sentimen
                else:
                    prob_sentimen_positif = 0

                st.write(f"**Probabilitas Sentimen Positif:** {round(prob_sentimen_positif * 100, 2)}%")

                # Kolom untuk GIF dan metric rating
                col1, col2, col3 = st.columns([1, 1, 1])

                # Rekomendasi berdasarkan probabilitas dan menampilkan GIF
                if prob_sentimen_positif == 1:
                    col1.success("Produk Direkomendasikan")
                    col1.image(gif_positive, use_column_width=True)  # Tampilkan GIF positif
                elif prob_sentimen_positif >= 0.5:
                    col1.info("Barang Rekomendasi")
                    col1.image(gif_positive, use_column_width=True)  # Tampilkan GIF positif
                else:
                    col1.error("Barang Tidak Direkomendasikan")
                    col1.image(gif_negative, use_column_width=True)  # Tampilkan GIF negatif

                # Perkiraan rating (sebagai contoh, Anda dapat menghitung nilai perkiraan berdasarkan sentimen atau data lain)
                rating = round(prob_sentimen_positif * 5, 2)  # Perkiraan rating dari 0 sampai 5
                col3.metric("Perkiraan Rating", rating, None)
            else:
                st.warning(f"Tidak ada ulasan untuk tipe barang: {tipe_barang_input}")
        else:
            st.error('Silakan pilih tipe barang untuk analisis.')

if __name__ == '__main__':
    main()
