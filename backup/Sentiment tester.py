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

    # Judul dan Pengantar
    st.title("Analisis Sentimen Ulasan Tokopedia")
    
    st.sidebar.title("Deteksi Bahasa dan Sentimen")
    
    # Input teks dari pengguna
    ulasan_input = st.sidebar.text_area("Masukkan ulasan Anda di sini", "")
    
    # Tombol untuk melakukan prediksi
    if st.sidebar.button("Prediksi"):
        # Preprocessing
        ulasan_preprocessed = preprocess_text(ulasan_input)
        deteksi_bahasa = detect_language(ulasan_preprocessed)
        
        # Prediksi Sentimen
        prediction_sentimen = modelsvc_loaded.predict([ulasan_preprocessed])[0]
        prediction_proba_sentimen = modelsvc_loaded.predict_proba([ulasan_preprocessed])[0][1]  # Probabilitas sentimen positif
        
        # Prediksi Tipe Barang
        prediction_tipe_barang = model_barang_loaded.predict([ulasan_preprocessed])[0]

        # Tentukan Layak atau Tidak
        if prediction_sentimen == 'positif':
            status_barang = "Layak Dibeli"
        else:
            status_barang = "Barang ini Tidak Layak Dibeli"

        # Tampilkan Hasil di Streamlit
        st.write(f"**Deteksi Bahasa:** {deteksi_bahasa}")
        st.write(f"**Tipe Barang:** {prediction_tipe_barang}")
        st.write(f"**Sentimen:** {prediction_sentimen}")
        st.write(f"**Probabilitas Sentimen Positif:** {round(prediction_proba_sentimen * 100, 2)}%")
        st.write(f"**Status Barang:** {status_barang}")
    
if __name__ == '__main__':
    main()
