import streamlit as st
import numpy as np
import cv2 as cv
import os
from io import BytesIO
import tensorflow as tf
from PIL import Image
import time

st.set_page_config(page_title="Klasifikasi", page_icon='')

page_bg_img = """
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

st.markdown(page_bg_img, unsafe_allow_html = True)
# st.markdown("<h5 style> = 'font")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_image = uploaded_file.read()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(" ")
    with col2:
        image  = Image.open(BytesIO(bytes_image))
        st.image(image, use_column_width = True)
    with col3:
        st.write(" ")

    image_np = np.asarray(image)
    resized_image = cv.resize(image_np, (256,256))