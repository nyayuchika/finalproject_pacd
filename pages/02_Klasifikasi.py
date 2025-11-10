import streamlit as st
import numpy as np
import cv2 as cv
import os
from io import BytesIO
from PIL import Image, ImageFilter
import time
import joblib

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
	#Grayscale
	gray_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
	#Gaussian Filter
	gaussian_image = cv.GaussianBlur(gray_image, (3,3), 0)
	#CLAHE
	clahe = cv.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
	clahe_image = clahe.apply(gaussian_image)
	
	#Otsu
	def otsu(image):
		#Membuat Histogram
		histogram = []
		for i in range(0, 256):
			nilai_awal = 0
			histogram.append(nilai_awal)
		#Hitung nilai intensitas tiap pixel
		for i in range(len(image)):
			for j in range(len(image[i])):
				nilai_pixel = image[i][j]
				histogram[nilai_pixel] = histogram[nilai_pixel] + 1
		#Hitung total pixel
		total = 0
		for i in range(256):
			total = total + histogram[i]
		best_threshold = 0
		min_within_class_var = float('inf')
		#Otsu dimulai
		for t in range (0, len(histogram)):
			#Background
			jumlah_histogram_b = 0
			for i in range(t):
				jumlah_histogram_b = jumlah_histogram_b + histogram[i]
			weight_b = jumlah_histogram_b / total
			jumlah_intensitas_b = 0
			jumlah_weight_b = 0
			for i in range(t):
				jumlah_intensitas_b = jumlah_intensitas_b + i * histogram[i]
				jumlah_weight_b = jumlah_weight_b + histogram[i]

			#Rata-rata
			if jumlah_weight_b == 0:
				mean_b = 0
			else:
				mean_b = jumlah_intensitas_b / jumlah_weight_b

			#Varians
			var_b = 0
			for i in range(t):
				var_b = var_b + ((i - mean_b) * (i - mean_b)) * histogram[i]
			if jumlah_weight_b != 0:
				var_b = var_b / jumlah_weight_b

			#------------------------------------------------------#

			#Foreground
			jumlah_histogram_f = 0
			for i in range(t, len(histogram)):
				jumlah_histogram_f = jumlah_histogram_f + histogram[i]
			weight_f = jumlah_histogram_f / total
			jumlah_intensitas_f = 0
			jumlah_weight_f = 0
			for i in range(t, len(histogram)):
				jumlah_intensitas_f = jumlah_intensitas_f + i * histogram[i]
				jumlah_weight_f = jumlah_weight_f + histogram[i]
			#Rata-rata
			if jumlah_weight_f == 0:
				mean_f = 0
			else:
				mean_f = jumlah_intensitas_f / jumlah_weight_f
			#Varians
			var_f = 0
			for i in range(t, len(histogram)):
				var_f = var_f + ((i - mean_f) * (i - mean_f)) * histogram[i]
			if jumlah_weight_f != 0:
				var_f = var_f / jumlah_weight_f

			#Within class variance
			within_class_var = (weight_b * var_b) + (weight_f * var_f)

			#Pilih nilai threshold minimum
			if within_class_var < min_within_class_var:
				min_within_class_var = within_class_var
			best_threshold = t
			print(f"t={t} | Wb={weight_b:.6f} | Wf={weight_f:.6f} | μb={mean_b:.6f} | μf={mean_f:.6f} | σ²w={within_class_var:.6f}")

		print("\nNilai threshold terbaik =", best_threshold)

		#Menerapkan otsu ke citra
		# otsu_image = (image >= best_threshold) * 255
		otsu_image = (image < best_threshold) * 255 #untuk hasil invers
		return otsu_image, best_threshold
	  
	mask, t_value = otsu(clahe_image)
	
	st.info(f"Nilai Threshold Otsu Terbaik: {t_value}")
	
	st.image(mask, caption = "Hasil Segmentasi Otsu", use_column_width = True)
	
	#Flood fill
	padded = cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT, value = 255)
	floodfill_mask = np.zeros((padded.shape[0]+2, padded.shape[1]+2), np.uint8)
	cv.floodFill(padded, floodfill_mask, (0,0), 0)
	flooded_image = padded[1:-1, 1:-1]
	
	
	flat_data = flooded_image.flatten()
	input_data = flat_data.reshape(1, -1)
	
	#Load Model
	svm_model = joblib.load('model/model_svm.pkl')
	
	#Prediksi
	result = svm_model.predict(input_data)
	with st.spinner('Loading...'):
		time.sleep(5)
	st.success('Done!')
	
	#Hasil prediksi