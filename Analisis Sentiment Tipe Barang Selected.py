import streamlit as st
import pickle
import re
import langdetect
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from PIL import Image
import time

# Set favicon
st.set_page_config(page_title="Analisis Sentiment Ulasan", page_icon="./images/code.png")

# Kamus Normalisasi tanpa spasi di awal dan akhir
norm = {
    'knp': 'kenapa', 'bgs': 'bagus', 'syg': 'sayang', 'bs': 'bisa', 'w': 'saya', 
    'muter': 'putar', 'ga': 'tidak', 'gk': 'tidak', 'brg': 'barang', 'sm': 'sama', 
    'dg': 'dengan', 'hrg': 'harga', 'g': 'tidak', 'yg': 'yang', 'ngak': 'tidak', 
    'jlk': 'jelek', 'mtp': 'mantap', 'krg': 'kurang', 'sdh': 'sudah', 'sy': 'saya', 
    'tdk': 'tidak', 'hr': 'hari', 'nga': 'tidak', 'kw': 'imitasi', 'ori': 'original', 
    'mantab': 'mantap', 'mayan': 'lumayan', 'tnx': 'terima kasih', 'jg': 'juga', 
    'gx': 'tidak', 'dlu': 'dulu', 'gak': 'tidak', 'tidk': 'tidak', 'ty': 'terima kasih', 
    'smg': 'semoga', 'pd': 'pada', 'tp': 'tapi', 'dtg': 'datang', 'pias': 'puas', 
    'jd': 'jadi', 'bgt': 'banget', 'cmn': 'cuman', 'barnag': 'barang', 'b': 'biasa', 
    'ngga': 'tidak', 'sip': 'baik', 'mantabs': 'mantap', 'asa': 'saja', 'aw': 'awet', 
    'rekomen': 'rekomendasi', 'mantapp': 'mantap', 'cui': 'teman', 'baguts': 'bagus', 
    'apik': 'baik', 'doi': 'dia', 'lom': 'belum', 'murmer': 'murah dan meriah', 
    'mantul': 'mantap betul', 'tksh': 'terima kasih', 'kwalitas': 'kualitas', 
    'mantaps': 'mantap', 'blm': 'belum', 'lbh': 'lebih', 'dr': 'dari', 
    'dgn': 'dengan', 'joss': 'mantap', 'josss': 'mantap', 'jos': 'mantap', 
    'manthab': 'mantap', 'ampe': 'sampai', 'ckp': 'cukup', 'gw': 'saya', 
    'bngt': 'banget', 'bwt': 'buat', 'zemoga': 'semoga', 'awets': 'awet', 
    'pqcking': 'paking', 'mntaoo': 'mantap', 'lg': 'lagi', 'krn': 'karena', 
    'mantappp': 'mantap', 'mantep': 'mantap', 'mangtap': 'mantap', 'bgtttt': 'banget', 
    'byk': 'banyak', 'cuku': 'cukup', 'n': 'dan', 'utk': 'untuk', 
    'tq': 'terima kasih', 'kualktas': 'kualitas', 'drskripsi': 'deskripsi', 
    'sgele': 'segel', 'lgsg': 'langsung', 'cepaf': 'cepat', 'dpf': 'dapat', 
    'brang': 'barang', 'ntaps': 'mantap', 'perfect': 'sempurna', 'smpai': 'sampai', 
    'ngk': 'tidak', 'dpt': 'dapat', 'perfecto': 'sempurna', 'flawless': 'sempurna', 
    'gmn': 'bagaimana', 'adl': 'adalah', 'emg': 'memang', 'manteppp': 'mantap', 
    'muantappp': 'mantap', 'mantapppppp': 'mantap', 'mantaaap': 'mantap', 
    'mantaap': 'mantap', 'mantaabb': 'mantap', 'manteb': 'mantap', 'skrng': 'sekarang', 
    'katany': 'katanya', 'mantulll': 'mantap', 'koq': 'kok', 'mantapppp': 'mantap', 
    'mantappppp': 'mantap', 'trmksh': 'terima kasih', 'peyot': 'penyok', 
    'ntapppp': 'mantap', 'msh': 'masih', 'mgkn': 'mungkin', 
    'muantapppp': 'mantap', 'muantap': 'mantap', 'muantaapppp': 'mantap', 
    'muantab': 'mantap', 'mantebbb': 'mantap', 'mantull': 'mantap', 
    'mantaaaap': 'mantap', 'mantaaaaappppppp': 'mantap', 'mantapss': 'mantap', 
    'bad': 'jelek', 'bkn': 'bukan', 'worst': 'buruk', 
    'd': 'deh', 'de': 'deh', 'good': 'mantap', 'topp': 'mantap', 
    'oriii': 'original', 'puassss': 'puas', 'mantpp': 'mantap', 
    'mksh': 'makasih', 'segellll': 'segel', 'oryginal': 'original', 
    'mantabbb': 'mantap', 'mantepp': 'mantap', 'ksh': 'kasih', 
    'seller': 'penjual', 'top': 'mantap', 'brp': 'berapa', 'mks': 'makasih', 'smpe': 'sampai',
    'cpt': 'cepat'
}

# Fungsi Normalisasi dengan Word Boundaries
def normalisasi(text):
    for key, value in norm.items():
        # Menggunakan \b untuk memastikan penggantian hanya pada kata yang tepat
        pattern = r'\b{}\b'.format(re.escape(key))
        text = re.sub(pattern, value, text)
    return text

# Load model dari file .pkl
with open('SVM_Pipe.pkl', 'rb') as model_file:
    modelsvc_loaded = pickle.load(model_file)

# Load model dari file .pkl untuk tipe barang
with open('SVM_PipeBarang.pkl', 'rb') as model_barang_file:
    model_barang_loaded = pickle.load(model_barang_file)

# Fungsi Preprocessing
def preprocess_text(text):
    # text = re.sub(r'\d+', '', text)  # Hapus angka
    text = re.sub(r'\W+', ' ', text)  # Hapus simbol dan karakter non-alfabet
    text = text.lower()  # Ubah menjadi huruf kecil
    text = normalisasi(text)  # Terapkan normalisasi
    stemmer = StemmerFactory().create_stemmer()  # Inisialisasi stemmer
    text = stemmer.stem(text)  # Proses stemming
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
    # Mengonversi probabilitas ke skala rating 1-5
    return round(proba * 5, 2)

# Fungsi Utama Halaman
def main(): 
    # Logo dan Gambar
    sen = './images/sent.png'
    sen1 = './images/sent.png'
    toped = './images/logotoped.png'
    lokam = './images/logo.png'
    sidebar_logo = sen
    main_body_logo = sen1
    st.logo(sidebar_logo, icon_image=main_body_logo)
    
    # Menampilkan Logo di Main Body
    col11, col12 = st.columns(2)
    col11.image(toped, width=170,)

    # Judul Aplikasi
    st.markdown("<h1 style='text-align: center; color: #50C878;'>Analisis Sentimen Ulasan Tokopedia</h1>", unsafe_allow_html=True)
    
    # Input Teks dari Pengguna
    userText = st.text_input('Masukkan Ulasan Produk:', placeholder='Paste Ulasan Disini..')
    
    # Tombol Analisis Sentimen
    if st.button('Analysis'):
        if userText:
            # Cek apakah input hanya satu kata
            if len(userText.split()) <= 1:
                st.toast('Mohon masukkan lebih dari satu kata untuk analisis sentimen.')
            else:
                # Tampilkan Ulasan Asli
                st.info(f"*Ulasan yang diinput:*\n\n{userText}")

                # Deteksi Bahasa
                lang = detect_language(userText)
                
                if lang == 'id':
                    # Preproses Teks
                    text_clean = preprocess_text(userText)
                    bar = st.progress(0)  # Initialize the progress bar
                    time.sleep(0.5)
                    bar.progress(50)  # Update to 50%
        
                    time.sleep(1)
                    bar.progress(100)  # Update to 100%
        
                    # Tampilkan Teks yang Telah Diproses
                    st.success(f"*Ulasan yang sudah diproses:*\n\n{text_clean}")
                    # Prediksi Tipe Barang
                    
                    # Transformasi Teks dengan Model dan Prediksi Sentimen
                    text_vector = modelsvc_loaded['vectorizer'].transform([text_clean])

                    # Prediksi Tipe Barang (setelah text_vector terdefinisi)
                    pred_tipe_barang = model_barang_loaded['classifier'].predict(text_vector)

                    # Prediksi Sentimen
                    prediction_proba = modelsvc_loaded['classifier'].predict_proba(text_vector)
                    proba_positif = prediction_proba[0][1]
                    
                    # Prediksi Rating
                    rating = predict_rating(proba_positif)
                    
                    # Tentukan Label Sentimen
                    sentiment_label = 'positif' if proba_positif >= 0.5 else 'negatif'
                    
                    # Tampilkan Hasil
                    st.markdown("<h3 style='color: #666699; font-family: Source Sans Pro, sans-serif; font-size: 25px; margin-bottom: 20px; margin-top: 50px;'>Hasil Analisis Sentimen Support Vector Machine:</strong></h3>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)

                    # Tampilkan GIF di col1
                    if sentiment_label == 'positif':
                        col1.image('./images/mots.gif', use_column_width=True, width=100)  # Tampilkan GIF positif
                    else:
                        col1.image('./images/motn.gif', use_column_width=True, width=100)   # Tampilkan GIF negatif

                    col3.metric("Perkiraan Rating", rating, None)
                    # Tampilkan tipe barang
                    col3.markdown(f"<h4>Tipe Barang: {pred_tipe_barang[0]}</h4>", unsafe_allow_html=True)
                    # Tampilkan Label Sentimen dengan Warna di col2
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
