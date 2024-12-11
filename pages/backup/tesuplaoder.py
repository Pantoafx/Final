import streamlit as st
import pandas as pd
import pickle

# Load model pickle (sesuaikan dengan file model kamu)
with open('SVM_Pipe.pkl', 'rb') as model_file:
    model_sentimen = pickle.load(model_file)

# Fungsi normalisasi
def normalisasi(text):
    normalisasi_dict = {
        'gk': 'tidak',
        'ga': 'tidak',
        'tdk': 'tidak',
        'sy': 'saya',
        'sm': 'sama',
        'dmn': 'dimana',
        'msh': 'masih',
        'trs': 'terus',
        'aja': 'saja',
        'bgt': 'banget',
        'pd': 'pada',
        'udh': 'sudah',
        'blm': 'belum',
        'jd': 'jadi',
        'dr': 'dari',
        'tp': 'tapi',
        'jg': 'juga',
        'klo': 'kalau',
        'kmrn': 'kemarin'
    }
    
    words = text.split()
    normalized_words = [normalisasi_dict[word] if word in normalisasi_dict else word for word in words]
    normalized_text = ' '.join(normalized_words)
    
    return normalized_text

# Fungsi untuk proses upload dan analisis
def analisis_data(uploaded_file):
    try:
        # Tampilkan isi file mentah (untuk debugging)
        st.write("Isi file mentah:")
        raw_data = uploaded_file.getvalue().decode("utf-8")
        st.text(raw_data)

        # Periksa apakah file kosong
        if raw_data.strip() == "":
            st.error("File CSV yang diunggah kosong.")
            return None

        # Coba berbagai jenis separator (menggunakan engine python)
        try:
            data = pd.read_csv(uploaded_file, sep=None, engine='python')
        except pd.errors.ParserError:
            st.warning("File tidak dapat diparsing dengan separator default, mencoba dengan separator ';'")
            data = pd.read_csv(uploaded_file, sep=';', engine='python')

        # Cek jika data kosong
        if data.empty or data.columns.size == 0:
            st.error("File CSV kosong atau tidak ada kolom yang dapat diproses.")
            return None

        # Normalisasi kolom ulasan
        if 'Ulasan' in data.columns:
            data['Ulasan_Normal'] = data['Ulasan'].apply(normalisasi)

            # Prediksi sentimen berdasarkan model SVM
            data['Prediksi_Sentimen'] = model_sentimen.predict(data['Ulasan_Normal'])

            # Hitung jumlah sentimen positif dan negatif
            jumlah_positif = sum(data['Prediksi_Sentimen'] == 'positif')
            jumlah_negatif = sum(data['Prediksi_Sentimen'] == 'negatif
