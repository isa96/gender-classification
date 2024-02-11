import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


### install streamlit terlebih dahulu -> pip install streamlit / conda install streamlit
### jalankan dengan streamlit run genderclf.py
# Fungsi preproses disesuaikan dengan model 
# jika terlalu panjang bisa diletakkan ke file terpisah
def proses_img(file):
    image = cv2.resize(plt.imread(file), (178, 218))
    image = np.expand_dims(np.array(image) / 255, axis=0)

    return image

# Load model, jangan lupa sesuaikan path agar tidak error
# Jika ingin mudah bisa diletakkan di folder yang sama
model = tf.keras.models.load_model('model.h5')

st.title("Pengenalan Gender")
file = st.file_uploader("Upload foto yang ingin diklasifikasi", type=['png','jpg','jpeg','webp'])
try:
    # menampilkan image yang diupload
    st.image(file, width=256)
except:
    st.write("Belum ada file dipuload")

if st.button('Klik untuk mengetahui gender'):
    try:
        image = proses_img(file)
        pred = model.predict(image)
        hasil = np.argmax(pred)
        if hasil == 1:
            st.write("**Gender teridentifikasi: laki-laki**")
        else:
            st.write("**Gender teridentifikasi: Perempuan**")
    except:
        st.write("UPLOAD FOTONYA DULU WOY!")

 