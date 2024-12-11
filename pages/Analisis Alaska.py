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
    
    # Tampilkan logo di sidebar
    st.sidebar.image(sidebar_logo, use_column_width=True)
    
    # Tampilkan logo di bagian utama
    st.image(main_body_logo, use_column_width=True)
    
    col11, col12 = st.columns(2)
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

           # Menampilkan hasil analisis dalam dua kolom
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h3 style='color: #666699; font-size: 25px;'>Hasil Berdasarkan Rating:</h3>", unsafe_allow_html=True)

            st.markdown(f"<span style='color: #ffa500;'>  Rating ★ (3-5):</span> {jumlah_positif_rating}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #ffa500;'>  Rating ★ (1-2):</span> {jumlah_negatif_rating}", unsafe_allow_html=True)
            # Menghitung probabilitas sentimen positif dari rating
            total_sentimen_rating = jumlah_positif_rating + jumlah_negatif_rating
            prob_sentimen_positif_rating = jumlah_positif_rating / total_sentimen_rating if total_sentimen_rating > 0 else 0

            st.write(f"**Probabilitas Sentimen Positif (Rating):** {round(prob_sentimen_positif_rating * 100, 2)}%")
        # Rekomendasi berdasarkan rating
            if prob_sentimen_positif_rating >= 0.5:
                st.success("Produk Direkomendasikan berdasarkan rating.")
            else:
                st.error("Produk Tidak Direkomendasikan berdasarkan rating.")

        with col2:
            
            st.markdown("<h3 style='color: #666699; font-size: 25px;'>Hasil Analisis SVM:</h3>", unsafe_allow_html=True)
            
            st.markdown(f"<span style='color: #50C878;'>Positif ✔️ (SVM):</span> {jumlah_positif_model}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #FF4B4B;'>Negatif ❌ (SVM):</span> {jumlah_negatif_model}", unsafe_allow_html=True)
             # Menghitung probabilitas sentimen positif dari model
            total_sentimen_model = jumlah_positif_model + jumlah_negatif_model
            prob_sentimen_positif_model = jumlah_positif_model / total_sentimen_model if total_sentimen_model > 0 else 0

            st.write(f"**Probabilitas Sentimen Positif (Model SVM):** {round(prob_sentimen_positif_model * 100, 2)}%")
            # Rekomendasi berdasarkan prediksi model
            if prob_sentimen_positif_model >= 0.5:
                st.success("Produk Direkomendasikan berdasarkan prediksi model.")
            else:
                st.error("Produk Tidak Direkomendasikan berdasarkan prediksi model.")

        
        
    else:
            st.info('Silakan pilih tipe barang untuk analisis.')

    # Tambahkan tombol untuk analisis keseluruhan
    if st.button('Analisis Keseluruhan'):
        # Prediksi sentimen untuk seluruh dataset
        df['Prediksi Sentimen'] = df['Ulasan'].apply(lambda x: predict_sentiment(x, modelsvc_loaded))

        # Menghitung jumlah sentimen positif dan negatif
        total_positif_model = len(df[df['Prediksi Sentimen'] == 'positif'])
        total_negatif_model = len(df[df['Prediksi Sentimen'] == 'negatif'])

        # Menghitung jumlah ulasan positif dan negatif berdasarkan rating
        total_positif_rating = len(df[df['Rating'] >= 3])
        total_negatif_rating = len(df[df['Rating'] < 3])

        st.markdown("<h3 style='color: #666699; font-size: 30px;'>Hasil Analisis Keseluruhan: <strong>Toko Alaska</strong></h3>", unsafe_allow_html=True)
        col11, col12 = st.columns(2)
        # Menampilkan hasil analisis keseluruhan

        with col11:
            st.markdown("Hasil Berdasarkan *Rating*")
            st.markdown(f"<span style='color:  #ffa500;'>Rating ★ (3-5):</span> {total_positif_rating}", unsafe_allow_html=True)
            st.markdown(f"<span style='color:  #ffa500;'>Rating ★ (1-2):</span> {total_negatif_rating}", unsafe_allow_html=True)
            # Menghitung probabilitas sentimen positif dari rating
            total_sentimen_rating_all = total_positif_rating + total_negatif_rating
            prob_sentimen_positif_rating_all = total_positif_rating / total_sentimen_rating_all if total_sentimen_rating_all > 0 else 0
            st.write(f"**Probabilitas Rating Positif dan Negatif (Rating):** {round(prob_sentimen_positif_rating_all * 100, 2)}%")
        
        with col12:
            st.markdown("Hasil Berdasarkan Analisis *(SVM)*")
            st.markdown(f"<span style='color: #50C878;'>Positif ✔️ (SVM):</span> {total_positif_model}", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #FF4B4B;'>Negatif ❌ (SVM):</span> {total_negatif_model}", unsafe_allow_html=True)
             # Menghitung probabilitas sentimen positif dari model
            total_sentimen_model_all = total_positif_model + total_negatif_model
            prob_sentimen_positif_model_all = total_positif_model / total_sentimen_model_all if total_sentimen_model_all > 0 else 0
            st.write(f"**Probabilitas Sentimen Analisis (SVM):** {round(prob_sentimen_positif_model_all * 100, 2)}%")
       
        # Kesimpulan untuk keseluruhan toko
        if prob_sentimen_positif_rating_all >= 0.5:
            st.success("Toko ini memiliki feedback bagus berdasarkan rating.")
        else:
            st.warning("Toko ini memiliki feedback kurang baik berdasarkan rating.")

        if prob_sentimen_positif_model_all >= 0.5:
            st.success("Toko ini memiliki feedback bagus berdasarkan prediksi model SVM.")
            st.info("Dari prediksi model (SVM) kebanyakan ulasan mengeluh terhadap *respon admin yang sangat lambat* dan *pengemasan barang kurang baik*")
        else:
            st.warning("Toko ini memiliki feedback kurang baik berdasarkan prediksi model.")

if __name__ == '__main__':
    main()
