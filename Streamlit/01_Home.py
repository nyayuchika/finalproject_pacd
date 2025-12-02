import streamlit as st
import requests
import json
from streamlit_lottie import st_lottie
import webbrowser

st.set_page_config(page_title='Home', layout='wide', page_icon='🏠')

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

st.markdown("""
<style>
    /* 1. Membuat tombol (st.button) rata tengah */
    div.stButton > button {
        display: block;
        margin: 0 auto;
    }

    /* 2. Membuat Lottie rata tengah */
    /* Menargetkan container Lottie di dalam Streamlit */
    div[data-testid="stLottie"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Opsional: Merapikan tampilan button agar lebih compact */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- DEFINISI KOLOM ---
col1, col2, col3, col4 = st.columns(4)

# Variable untuk margin negatif (ubah angka ini untuk mengatur jarak Lottie ke Teks)
# Semakin besar angkanya (misal -40px), teks semakin naik ke atas.
with container:
	st.markdown("<h2 style='font-family:sans-serif; text-align:center; margin:0; padding-top:0;'>Kelompok 1</h2>", unsafe_allow_html=True)
	text_margin_top = "-30px" 
	with col1:
		lottie_chika = "https://lottie.host/34a80f3f-1f13-4a87-b01d-55f0c0e51312/D1FbbNLpk0.json"
		chika_json = load_lottieurl(lottie_chika)
	
		st_lottie(chika_json, height=200, width=200, key="chika")
		
		st.markdown(f"<h5 style='font-family:sans-serif; text-align:center; margin-bottom:0; margin-top:{text_margin_top}; font-size:16px;'>Nyayu Chika Marselina**</h5>", unsafe_allow_html=True)
		st.markdown("<h5 style='font-family:sans-serif; text-align:center; margin:0; padding-top:0; font-weight:normal; font-size:14px;'>25/568182/PPA/07148</h5>", unsafe_allow_html=True)
		
		if st.button('Github Chika', use_container_width=True):
			webbrowser.open_new_tab('https://github.com/nyayuchika')

	with col2:
		lottie_kak_tiwi = "https://lottie.host/1e4cda4f-06cc-40d8-bf15-8e897133c0b6/O49pFQJ3Ov.json"
		kak_tiwi_json = load_lottieurl(lottie_kak_tiwi)
		
		st_lottie(kak_tiwi_json, height=200, width=200, key="tiwi")
		
		st.markdown(f"<h5 style='font-family:sans-serif; text-align:center; margin-bottom:0; margin-top:{text_margin_top}; font-size:16px;'>Yullase Pratiwi**</h5>", unsafe_allow_html=True)
		st.markdown("<h5 style='font-family:sans-serif; text-align:center; margin:0; padding-top:0; font-weight:normal; font-size:14px;'>24/550766/PPA/06955</h5>", unsafe_allow_html=True)
		
		if st.button('Github Kak Tiwi', use_container_width=True):
			webbrowser.open_new_tab('https://github.com/yullasepratiwi')

	with col3:
		lottie_fawwaz = "https://lottie.host/a7f60890-fb45-420f-a493-44eb34b1e15f/P8lFYtD9ZU.json"
		fawwaz_json = load_lottieurl(lottie_fawwaz)
		
		st_lottie(fawwaz_json, height=200, width=200, key="fawwaz")
		
		st.markdown(f"<h5 style='font-family:sans-serif; text-align:center; margin-bottom:0; margin-top:{text_margin_top}; font-size:16px;'>Fawwaz Rif’at Revista**</h5>", unsafe_allow_html=True)
		st.markdown("<h5 style='font-family:sans-serif; text-align:center; margin:0; padding-top:0; font-weight:normal; font-size:14px;'>25/565782/PPA/07130</h5>", unsafe_allow_html=True)
		
		if st.button('Github Fawwaz', use_container_width=True):
			webbrowser.open_new_tab('https://github.com/Fawwaz1233')

	with col4:
		lottie_mas_galih = "https://lottie.host/b8915619-3b59-446b-b30c-cc377f1c87ec/nycnrrcH72.json"
		mas_galih_json = load_lottieurl(lottie_mas_galih)
		
		st_lottie(mas_galih_json, height=200, width=200, key="galih")
		
		st.markdown(f"<h5 style='font-family:sans-serif; text-align:center; margin-bottom:0; margin-top:{text_margin_top}; font-size:16px;'>Galih Prabasidi**</h5>", unsafe_allow_html=True)
		st.markdown("<h5 style='font-family:sans-serif; text-align:center; margin:0; padding-top:0; font-weight:normal; font-size:14px;'>25/563010/PPA/07092</h5>", unsafe_allow_html=True)
		
		if st.button('Github Mas Galih', use_container_width=True):
			webbrowser.open_new_tab('https://github.com/UsagiUG')