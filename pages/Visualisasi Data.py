import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import time

# Set favicon
st.set_page_config(page_title="Analisis Sentiment Ulasan", page_icon="./images/code.png")

vis = './images/piechart.png'
vis1 = './images/piechart.png'
sidebar_logo = vis 
main_body_logo = vis1
st.logo(sidebar_logo, icon_image=main_body_logo)

with st.spinner(text='Load Data ...'):
        time.sleep(3)  # Waktu simulasi proses loading

vis = './images/piechart.png'
vis1 = './images/piechart.png'
sidebar_logo = vis 
main_body_logo = vis1
st.logo(sidebar_logo, icon_image=main_body_logo)

# Sidebar untuk kesimpulan
with st.sidebar:
    st.info("""
    **Kesimpulan:**
    
    Analisis menunjukkan bahwa mayoritas ulasan di Tokopedia memiliki **rating 5**, menunjukkan kepuasan pengguna yang tinggi. Dari 30 **produk terlaris**, sebagian besar ulasan bersifat **positif** dengan variasi panjang ulasan yang cukup beragam. Kata-kata kunci yang dominan berhasil diidentifikasi untuk ulasan **positif** dan **negatif**. Model machine learning yang digunakan mencapai **akurasi 88.67%** dengan nilai **AUC 0.94**, menunjukkan performa yang sangat baik dalam memprediksi sentimen.
    """)


st.markdown("<h1 style='text-align: center; color: #ffa500;'>Ringkasan Visualisasi Data</h1>", unsafe_allow_html=True)
st.markdown("Library Yang digunakan pada aplikasi ini")
st.code("""
# Import libraries dasar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize
# Unduh model tokenisasi nltk
nltk.download('punkt')

# Import libraries untuk Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Import libraries lainnya
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
import plotly.express as px
import pickle
""", language='python')

st.caption("**Note:** Visualisasi singkat ini berupa dataframe dan hasil visual, tidak script programnya")
# Load Data set
df = pd.read_csv('DataScrappingAug.csv')

# Tampilkan DataFrame
st.dataframe(df) 


# Menghitung jumlah rating
rating_counts = df['Rating'].value_counts().reset_index()
rating_counts.columns = ['Rating', 'Jumlah']
st.caption("Data yang digunakan dalam analisis sentimen ini merupakan (Data Primer) yang dikumpulkan oleh peneliti. Data ini berupa teks ulasan pengguna yang diambil dari platform Tokopedia. Proses pengumpulan data dilakukan melalui teknik web scraping, di mana ulasan-ulasan tersebut diekstraksi secara langsung dari halaman ulasan di toko official di Tokopedia. Setelah melakukan scrapping data bulan Juli 2024 hingga Agustus 2024")
st.write("")  # 
st.markdown("<br>", unsafe_allow_html=True)

# Membuat barchart untuk rating
fig = px.bar(rating_counts, x='Rating', y='Jumlah', title='Jumlah Rating', labels={'Rating': 'Rating', 'Jumlah': 'Jumlah'})
st.plotly_chart(fig)
st.caption("Visualisasi distribusi rating yang diberikan oleh pengguna terhadap produk di Tokopedia. Distribusi ini dimulai dari rating terendah hingga rating tertinggi, dengan jumlah ulasan terbanyak berada pada rating 5. Dari grafik tersebut, dapat dilihat bahwa sebagian besar pengguna memberikan rating yang lebih tinggi, menunjukkan tingkat kepuasan yang relatif baik terhadap produk yang diulas. Sebaliknya, rating terendah memiliki jumlah ulasan yang lebih sedikit, yang mengindikasikan bahwa hanya sebagian kecil pengguna yang merasa kurang puas. Distribusi ini memberikan gambaran awal tentang persepsi umum pengguna terhadap produk dan menjadi dasar untuk analisis")
st.write("")  # 
st.markdown("<br>", unsafe_allow_html=True)

# Menghitung tipe barang paling laris
# Menghitung tipe barang paling laris
barang_terlaris = df['Tipe Barang'].value_counts().reset_index()
barang_terlaris.columns = ['Tipe Barang', 'Jumlah']

# Menampilkan hanya 30 tipe barang paling laris
barang_terlaris_top30 = barang_terlaris.head(30)

# Membuat barchart horizontal untuk tipe barang paling laris
fig_barang = px.bar(barang_terlaris_top30,
                    x='Jumlah', 
                    y='Tipe Barang', 
                    title='30 Tipe Barang Paling Laris', 
                    labels={'Tipe Barang': 'Tipe Barang', 'Jumlah': 'Jumlah'},
                    orientation='h',  # Membuat barchart horizontal
                    color='Jumlah',  # Menambahkan warna berdasarkan jumlah
                    color_continuous_scale=px.colors.sequential.Viridis)

# Menampilkan barchart di Streamlit
st.plotly_chart(fig_barang)
st.caption("Visualisasi 30 Tipe Barang Paling Laris Dalam Bentuk barchart Horizontal")
st.write("")  # 
st.markdown("<br>", unsafe_allow_html=True)

# Menampilkan hanya 30 tipe barang paling laris
barang_terlaris_top30 = barang_terlaris.head(30)

# Membuat scatter plot untuk tipe barang paling laris
fig_scatter = px.scatter(barang_terlaris_top30,
                          x='Jumlah', 
                          y=barang_terlaris_top30.index,  # Menggunakan indeks sebagai sumbu y
                          title='30 Tipe Barang Paling Laris',
                          labels={'y': 'Tipe Barang'},
                          hover_name='Tipe Barang',  # Menampilkan nama tipe barang saat hover
                          color='Jumlah',  # Menambahkan warna berdasarkan jumlah
                          color_continuous_scale=px.colors.sequential.Viridis)

# Menampilkan scatter plot di Streamlit
st.plotly_chart(fig_scatter)
st.caption("Visualisasi 30 Tipe Barang Paling Laris Dalam Bentuk Scatter Plot")
st.write("")  # 
st.markdown("<br>", unsafe_allow_html=True)


df = pd.read_csv('DataBersihTokopedia.csv')
# Tampilkan DataFrame
st.dataframe(df)

st.caption("""
Tampilan diatas menampilkan pengambilan data dari tabel Ulasan dan Rating. 
Pada tahap ini, hanya dua atribut utama yang akan dikelola, yaitu ulasan dan rating.<br>
""", unsafe_allow_html=True)
st.toast("Note: beberapa tahapan tidak akan dijelaskan lebih detail.")

st.write("")  # 
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<h5>Dataframe Bersih yang sudah di labeling dan fungsi deteksi bahasa sudah di implentasikan </h5>", unsafe_allow_html=True)

df = pd.read_csv('DataBersihSentimen.csv')
# Tampilkan DataFrame
st.dataframe(df)

st.write("")  # 
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<h5>Visualisasi Barchart dan lain - lain </h5>", unsafe_allow_html=True)

# Menghitung jumlah sentimen positif dan negatif
sentiment_counts = df['Sentimen'].value_counts().reset_index()
sentiment_counts.columns = ['Sentimen', 'Count']

# Membuat barchart
fig_bar = px.bar(sentiment_counts, x='Sentimen', y='Count', title='Barchart Sentimen Positif dan Negatif')

# Menampilkan barchart
st.plotly_chart(fig_bar)

fig_pie = px.pie(sentiment_counts, names='Sentimen', values='Count', title='Pie Chart Sentimen Positif dan Negatif')

# Menampilkan pie chart
st.plotly_chart(fig_pie)

# Menghitung panjang ulasan
df['Panjang_Ulasan'] = df['Ulasan'].apply(len)

# Membuat histogram distribusi panjang ulasan
fig_hist = px.histogram(df, x='Panjang_Ulasan', title='Distribusi Panjang Ulasan', nbins=30)

# Menampilkan histogram
st.plotly_chart(fig_hist)

# Memisahkan ulasan positif dan negatif
positive_reviews = df[df['Sentimen'] == 'positif']['Ulasan']
negative_reviews = df[df['Sentimen'] == 'negatif']['Ulasan']

# Menghitung frekuensi kata untuk ulasan positif
positive_words = ' '.join(positive_reviews).lower().split()
negative_words = ' '.join(negative_reviews).lower().split()

# Menghilangkan stopwords
stop_words = set(stopwords.words('indonesian'))
positive_words = [word for word in positive_words if word not in stop_words]
negative_words = [word for word in negative_words if word not in stop_words]

# Menghitung frekuensi kata
positive_word_freq = Counter(positive_words).most_common(50)
negative_word_freq = Counter(negative_words).most_common(50)

# Mengonversi frekuensi kata menjadi DataFrame
positive_df = pd.DataFrame(positive_word_freq, columns=['Kata', 'Frekuensi'])
negative_df = pd.DataFrame(negative_word_freq, columns=['Kata', 'Frekuensi'])

# Membuat barchart untuk kata positif
fig_pos = px.bar(positive_df, x='Kata', y='Frekuensi', title='50 Frekuensi Kata Positif', color='Frekuensi', text='Frekuensi')
fig_pos.update_traces(texttemplate='%{text}', textposition='outside')
fig_pos.update_layout(xaxis_title='Kata', yaxis_title='Frekuensi', xaxis_tickangle=-45)

# Menampilkan barchart kata positif
st.plotly_chart(fig_pos)

# Membuat barchart untuk kata negatif
fig_neg = px.bar(negative_df, x='Kata', y='Frekuensi', title='50 Frekuensi Kata Negatif', color='Frekuensi', text='Frekuensi')
fig_neg.update_traces(texttemplate='%{text}', textposition='outside')
fig_neg.update_layout(xaxis_title='Kata', yaxis_title='Frekuensi', xaxis_tickangle=-45)

# Menampilkan barchart kata negatif
st.plotly_chart(fig_neg)


st.image('./images/wcpositif.png')

st.write("")  # 
st.markdown("<br>", unsafe_allow_html=True)

st.image('./images/wcnegatif.png')

st.write("")  # 
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<h5> Visualisasi Hasil Pengujian Confusion Matrix Dan ROC </h5>", unsafe_allow_html=True)

st.image('./images/cm.png')
st.caption("Secara keseluruhan, model ini memiliki akurasi yang cukup baik (88.67%), dengan presisi dan recall yang tinggi, menunjukkan bahwa model ini mampu memprediksi positif dengan tepat serta menangkap sebagian besar data positif.")

st.write("")  # 
st.markdown("<br>", unsafe_allow_html=True)

st.image('./images/roc.png')

st.caption("Kurva ROC (Receiver Operating Characteristic) menunjukkan hubungan antara True Positive Rate (TPR) atau sensitivitas dengan False Positive Rate (FPR) pada berbagai threshold klasifikasi. AUC (Area Under the Curve) adalah 0.94, yang berarti model Anda memiliki performa yang sangat baik. Nilai AUC berkisar antara 0.5 (klasifikasi acak) hingga 1 (klasifikasi sempurna). Dengan nilai AUC sebesar 0.94, model ini memiliki kemampuan yang sangat baik dalam membedakan antara kelas positif dan negatif. Semakin dekat nilai AUC ke 1, semakin baik model dalam memprediksi hasil yang benar")