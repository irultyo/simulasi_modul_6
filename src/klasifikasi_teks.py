import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
import joblib

st.title("Klasifikasi Teks")
text = st.text_input("Teks")

def prediction():
    tokenizer = joblib.load(Path(__file__).parent / 
        "model/tokenizer.joblib")
    model = tf.keras.models.load_model(Path(__file__).parent / 
        "model/teks.h5")
    sequences = tokenizer.texts_to_sequences(text)
    pad_seq = tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=10, 
        padding='post')
    result = (model.predict(pad_seq) > 0.5).astype("int32")
    return result[0][0]

if st.button("Prediksi", type="primary"):
    st.subheader("Hasil prediksi: ")
    classes = ["negatif", "positif"]
    with st.spinner('Memproses teks untuk prediksi..'):
        result = prediction()
    st.write("Teks: "+text)
    st.subheader("Hasil prediksi: ")
    st.write(classes[result])