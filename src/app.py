import streamlit as st

pg = st.navigation([
    st.Page("klasifikasi_tabular.py", title="Klasifikasi Tabular"),
    st.Page("klasifikasi_citra.py", title="Klasifikasi Citra"),
    st.Page("klasifikasi_teks.py", title="Klasifikasi Teks"),
    ])
pg.run()