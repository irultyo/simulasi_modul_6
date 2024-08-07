import streamlit as st
import tensorflow as tf
from pathlib import Path
import numpy as np

st.title("Klasifikasi Citra")
upload = st.file_uploader(
    'Unggah citra untuk mendapatkan hasil prediksi', 
    type=['png','jpg'])

def predict():
    class_names = ["peach", "pomegranate", "strawberry"]
    
    img = tf.keras.utils.load_img(upload, target_size=(300, 300))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    model = tf.keras.models.load_model(Path(__file__).parent / 
    "model/citra.h5")
    output = model.predict(img_array)
    score = tf.nn.softmax(output[0])
    return class_names[np.argmax(score)]

if st.button("Predict", type="primary"):
    if upload is not None:
        st.image(upload)
        st.subheader("Hasil prediksi: ")
        with st.spinner('Memproses citra untuk prediksi..'):
            result = predict()
        st.write(result) 
    else:
        st.write("Unggah citra terlebih dahulu!!")