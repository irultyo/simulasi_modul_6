import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
import joblib

st.title("Klasifikasi Tabular")
f1 = st.slider("Fitur 1", -1.0, 2.5)
f2 = st.slider("Fitur 2", -1.0, 1.5)

def prediction():
    scaler = joblib.load(Path(__file__).parent / "model/scaler.joblib")
    model = tf.keras.models.load_model(Path(__file__).parent /
     "model/tabular.h5")
    data = scaler.transform([[f1,f2]])
    result = (model.predict(data) > 0.5).astype("int32")
    return result[0][0]

if st.button("Prediksi", type="primary"):
    st.subheader("Hasil prediksi: ")
    classes = ["Label 1", "Label 2"]
    with st.spinner('Memproses data untuk prediksi..'):
        result = prediction()
    st.write(classes[result])