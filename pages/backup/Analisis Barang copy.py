import streamlit as st
import pandas as pd
import re
import pickle
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
df = pd.read_csv('Datalaska.csv')

# Fungsi utama halaman
def main():
    # Menampilkan logo dan gambar
    inven = './images/invent.png'
    toped = './images/logotoped.png'
    sidebar_logo = inven
    main_body_logo = inven
    gif_positive = './images/mots.gif'
    gif_negative = './images/motn.gif'
    
    col11, col12, col13 = st.columns(3)
    col11.image(toped, use_column_width=True)

    # Sidebar
    st.sidebar.image(sidebar_logo)
    
    st.markdown("<h1 style='text-align: center; color: #50C878;'>Analisis Tipe Barang Tokopedia</h1>", unsafe_allow_html=True)
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

            # Menghitung prediksi sentimen menggunakan model yang dimuat dari file pkl
            filtered_df['Prediksi Sentimen'] = filtered_df['Ulasan'].apply(lambda x: predict_sentiment(x, modelsvc_loaded))

            # Menghitung jumlah sentimen positif dan negatif
            jumlah_positif = len(filtered_df[filtered_df['Prediksi Sentimen'] == 'positif'])
            jumlah_negatif = len(filtered_df[filtered_df['Prediksi Sentimen'] == 'negatif'])

            # Menampilkan hasil analisis
            st.markdown("<h3 style='color: #666699; font-size: 25px;'>Hasil Analisis Tipe Barang:</h3>", unsafe_allow_html=True)
            st.markdown(f"**Jumlah Sentimen untuk produk <strong style='color:#22dd88;'>{tipe_barang_input}</strong>**", unsafe_allow_html=True)
            st.markdown(f"<span style='color: green;'>Positif:</span> {jumlah_positif}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: red;'>Negatif:</span> {jumlah_negatif}", unsafe_allow_html=True)

            # Menghitung probabilitas sentimen positif
            total_sentimen = jumlah_positif + jumlah_negatif
            prob_sentimen_positif = jumlah_positif / total_sentimen if total_sentimen > 0 else 0

            st.write(f"**Probabilitas Sentimen Positif:** {round(prob_sentimen_positif * 100, 2)}%")

            # Kolom untuk GIF dan metric rating
            col1, col2, col3 = st.columns([1, 1, 1])

            # Rekomendasi berdasarkan probabilitas dan menampilkan GIF
            if prob_sentimen_positif == 1:
                col1.success("Produk Direkomendasikan")
                col1.image(gif_positive, use_column_width=True, width=100)  # Tampilkan GIF positif
            elif prob_sentimen_positif >= 0.5:
                col1.info("Barang Rekomendasi")
                col1.image(gif_positive, use_column_width=True, width=100)  # Tampilkan GIF positif
            else:
                col1.error("Barang Tidak Direkomendasikan")
                col1.image(gif_negative, use_column_width=True, width=100)  # Tampilkan GIF negatif

            # Perkiraan rating (sebagai contoh, Anda dapat menghitung nilai perkiraan berdasarkan sentimen atau data lain)
            rating = round(prob_sentimen_positif * 5, 2)  # Perkiraan rating dari 0 sampai 5
            col3.metric("Perkiraan Rating", rating, None)

            # Deteksi ulasan positif yang berisi kritik
            kata_kritik = ['buruk', 'tidak puas', 'mengecewakan', 'jelek', 'keluhan', 'lama', 'cacat', 'tp', 'tapi']

            # Fungsi untuk mendeteksi apakah ulasan mengandung kritik
            def contains_criticism(text, kata_kritik):
                return any(critique in text.lower() for critique in kata_kritik)

            bintang_5_kritik = filtered_df[
                (filtered_df['Prediksi Sentimen'] == 'positif') &
                (filtered_df['Ulasan'].apply(lambda x: contains_criticism(x, kata_kritik)))
            ]

            st.markdown("### Ulasan Positif yang Berisi Kritik")
            st.write(bintang_5_kritik[['Ulasan', 'Prediksi Sentimen']])

            # Kesimpulan total sentimen untuk semua tipe barang
            total_jumlah_positif = len(df[df['Ulasan'].apply(lambda x: predict_sentiment(x, modelsvc_loaded) == 'positif')])
            total_jumlah_negatif = len(df[df['Ulasan'].apply(lambda x: predict_sentiment(x, modelsvc_loaded) == 'negatif')])

            # Menampilkan kesimpulan
            st.markdown("### Kesimpulan Keseluruhan")
            total_sentimen_all = total_jumlah_positif + total_jumlah_negatif
            if total_sentimen_all > 0:
                prob_sentimen_positif_all = total_jumlah_positif / total_sentimen_all
                st.write(f"**Jumlah Ulasan Positif:** {total_jumlah_positif}")
                st.write(f"**Jumlah Ulasan Negatif:** {total_jumlah_negatif}")
                st.write(f"**Probabilitas Sentimen Positif (Keseluruhan):** {round(prob_sentimen_positif_all * 100, 2)}%")
                if prob_sentimen_positif_all >= 0.5:
                    st.success("Toko ini memberikan umpan balik dari konsumen, rata-rata merasa puas.")
                else:
                    st.warning("Toko ini memberikan umpan balik dari konsumen, rata-rata tidak puas.")
            else:
                st.warning("Tidak ada ulasan untuk dianalisis.")

        else:
            st.error('Silakan pilih tipe barang untuk analisis.')

if __name__ == '__main__':
    main()
