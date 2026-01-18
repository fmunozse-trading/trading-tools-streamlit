import streamlit as st

st.set_page_config(
    page_title="Hello World App",
    page_icon="👋",
)

import utils
utils.apply_global_styles()

st.title("Hello World! 👋")

st.write("Usa la barra lateral para navegar entre las diferentes herramientas.")
