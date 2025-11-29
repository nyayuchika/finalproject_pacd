import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import requests
import json
from streamlit_lottie import st_lottie

st.set_page_config(page_title='Home', layout='wide')

page_bg = """
<style>
[data-testid = "stAppViewContainer"]{
    background-color: #FFFFF;
    }
[data-testid = "stHeader"]{
    background-color: #008CFF;
    }
[data-testid = "stSidebar"]{
background-color: #B8E7FF}
</style>
"""

st.markdown(page_bg, unsafe_allow_html = True)

container = st.container()
with container:
    home1, home2 = st.columns([11,8])
    with home1:
        st.title("Klasifikasi Penyakit Tuberkulosis pada Citra X-Ray Thorax")
        st.subheader("Unggah hasil scan Anda dan dapatkan hasilnya.")
        if st.button("Get Started"):
            st.switch_page("pages/02_Klasifikasi.py")
    with home2:
        @st.cache_data
        def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()
        lottie_url = "https://lottie.host/fbbe1e78-b6b6-4ba1-9e9d-4d13d65266cb/lSkzfjkwTS.json"
        lottie_json = load_lottieurl(lottie_url)
        st_lottie(lottie_json, height=400)