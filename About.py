import streamlit as st
import pandas as pd
from PIL import Image

# Set favicon
st.set_page_config(page_title="Analisis Sentiment Ulasan", page_icon="./images/code.png")


#Load Data set
abot = './images/logo.png'
abot1 = './images/logo.png'
sidebar_logo = abot 
main_body_logo = abot1
st.logo(sidebar_logo, icon_image=main_body_logo)
st.sidebar.markdown(
    '<div style="text-align: center;"><a href="https://handayani.ac.id/" style="font-size:20px; text-decoration:none;" target="_blank">Universitas Handayani Makassar</a></div>',
    unsafe_allow_html=True
)
st.sidebar.write("")
st.sidebar.info(
    """
    **Readmee:**

    - Aplikasi ini dikembangkan di Jupyter Notebook
    - Analisis Sentiment pada sidebar sudah terintegrasi dalam bentuk pipeline, sehingga proses analisis berjalan otomatis dari awal hingga akhir.
    
    """
)


# st.sidebar.write("")
# st.sidebar.write("**Readmeee:**")
# st.sidebar.write("Aplikasi ini dikembangkan di Jupyter Notebook.")
# st.sidebar.write("Analisis Sentiment pada sidebar sudah terintegrasi dalam bentuk pipeline, sehingga proses analisis berjalan otomatis dari awal hingga akhir.")

# st.sidebar.write("")
# st.sidebar.markdown(
#     """
#     <div style="text-align: justify;">
#         <strong>Readmeee:</strong>
#         <ul>
#             <li>Aplikasi ini dikembangkan di Jupyter Notebook.</li>
#             <li>Analisis Sentiment pada sidebar sudah terintegrasi dalam bentuk pipeline, sehingga proses analisis berjalan otomatis dari awal hingga akhir.</li>
#         </ul>
#     </div>
#     """, 
#     unsafe_allow_html=True
# )


st.write("""
<style>
    .justify {
        text-align: justify;
    }
</style>
<div class="justify">
    <h3>Penjelasan Cara Kerja Aplikasi Analisis Sentiment</h3>
    <p>Aplikasi analisis sentimen yang dibangun menggunakan Streamlit dirancang untuk memberikan pengalaman pengguna yang intuitif dan responsif. Dengan memanfaatkan model <strong>Support Vector Machine (SVM)</strong> yang telah dilatih dan disimpan dalam format .pkl, aplikasi ini mampu melakukan analisis sentimen secara <strong>real-time</strong>.</p>
    <p>Untuk memulai, silakan ambil ulasan dari <a href="https://www.tokopedia.com" target="_blank" style="color: #50C878; text-decoration:none;">Tokopedia</a> dan paste ulasan tersebut di sidebar <strong>Analisis Sentiment</strong> untuk dianalisis.</p>
    <p>Proses analisis dimulai dengan memuat model SVM beserta pipeline-nya. Aplikasi secara otomatis memproses teks ulasan yang dimasukkan, termasuk langkah-langkah seperti preprocessing, deteksi bahasa, dan prediksi sentimen. Hasil analisis ini akan ditampilkan dengan visualisasi yang membantu pengguna memahami sentimen dan perkiraan rating. Pendekatan ini memungkinkan aplikasi memberikan informasi yang berguna dan relevan dalam waktu singkat.</p>
</div>
""", unsafe_allow_html=True)





st.caption("Aplikasi ini dibuat untuk memenuhi syarat untuk mendapatkan gelar **S1** di ***Universitas Handayani Makassar***")
