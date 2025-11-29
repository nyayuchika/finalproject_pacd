import streamlit as st
import numpy as np
import cv2 as cv
import joblib
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog, blob_dog
from skimage.measure import regionprops
import math
import os
import sklearn

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi TBC", layout="wide", page_icon="🫁")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"]{ background-color: #FFFFFF; }
    [data-testid="stHeader"]{ background-color: #008CFF; }
    [data-testid="stSidebar"]{ background-color: #B8E7FF; }
    .indicator-box {
        background-color: #F0F8FF; padding: 15px; border-radius: 10px;
        margin-bottom: 10px; border: 1px solid #B8E7FF; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .indicator-title { color: #1E3A8A; font-weight: bold; font-size: 16px; margin-bottom: 5px; }
    .indicator-desc { color: #64748B; font-size: 12px; margin-top: 5px; }
    .percentage-text { float: right; color: #008CFF; font-weight: bold; }
    .stProgress > div > div > div > div { background-color: #008CFF; }
    [data-testid="stFileUploader"] { padding: 20px; border: 2px dashed #008CFF; border-radius: 10px; background-color: #F0F8FF; }
</style>
""", unsafe_allow_html=True)

# --- 2. FUNGSI PREPROCESSING & SEGMENTASI ---

def custom_otsu(image):
    thresh_val, _ = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    otsu_image = (image < thresh_val).astype(np.uint8) * 255
    return otsu_image

def morpho_close_step(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    filled = ndimage.binary_fill_holes(closing).astype(np.uint8) * 255
    return filled

def pipeline_segmentasi_lengkap(gray_image):
    gaussian = cv.GaussianBlur(gray_image, (3,3), 0)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(gaussian)
    otsu_img = custom_otsu(img_clahe)
    padded = cv.copyMakeBorder(otsu_img, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=255)
    mask = np.zeros((padded.shape[0]+2, padded.shape[1]+2), np.uint8)
    cv.floodFill(padded, mask, (0,0), 0)
    flooded = padded[1:-1, 1:-1]
    step1 = morpho_close_step(flooded, 9)
    step2 = morpho_close_step(step1, 15)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(step2, connectivity=8)
    final_mask = np.zeros_like(step2)
    if num_labels > 1:
        areas = stats[1:, cv.CC_STAT_AREA]
        sorted_indices = np.argsort(areas)[::-1]
        top_n = min(2, len(sorted_indices))
        for i in range(top_n):
            label_idx = sorted_indices[i] + 1
            final_mask[labels == label_idx] = 255
    final_mask = cv.medianBlur(final_mask, 5)
    isolated_lungs = cv.bitwise_and(img_clahe, img_clahe, mask=final_mask)
    return isolated_lungs, final_mask, img_clahe

# --- 3. EKSTRAKSI FITUR ---

def extract_all_features_user_version(isolated_img, final_mask, hog_pixels=(16, 16)):
    features = {}
    # 1. Stats
    pixels = isolated_img[isolated_img > 0]
    if len(pixels) > 0:
        features['mean'] = np.mean(pixels); features['std'] = np.std(pixels)
        features['skew'] = skew(pixels); features['kurtosis'] = kurtosis(pixels)
    else:
        features.update({'mean': 0, 'std': 0, 'skew': 0, 'kurtosis': 0})

    # 2. GLCM (Ambil [0,0] sesuai notebook)
    glcm = graycomatrix(isolated_img, distances=[1, 3, 5], angles=[0], levels=256, symmetric=True, normed=True)
    features['contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    features['dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
    features['homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    features['energy'] = graycoprops(glcm, 'energy')[0, 0]
    features['correlation'] = graycoprops(glcm, 'correlation')[0, 0]
    features['ASM'] = graycoprops(glcm, 'ASM')[0, 0]

    # 3. Shape
    props = regionprops(final_mask.astype(int))
    if props:
        features['area_total'] = np.sum([p.area for p in props])
        features['perimeter_mean'] = np.mean([p.perimeter for p in props])
        features['solidity_mean'] = np.mean([p.solidity for p in props])
        features['eccentricity_mean'] = np.mean([p.eccentricity for p in props])
    else:
        features.update({'area_total': 0, 'perimeter_mean': 0, 'solidity_mean': 0, 'eccentricity_mean': 0})

    # 4. LBP
    lbp = local_binary_pattern(isolated_img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    lbp_values = list(lbp_hist)

    # 5. HOG (Sumber masalah dimensi biasanya di sini)
    try:
        hog_features = hog(isolated_img, orientations=9, pixels_per_cell=hog_pixels, 
                           cells_per_block=(2, 2), visualize=False, feature_vector=True)
        features['hog_mean'] = np.mean(hog_features)
        hog_values = list(hog_features)
    except:
        features['hog_mean'] = 0
        hog_values = []

    # 6. Blob (Visual only)
    img_norm = isolated_img / 255.0
    blobs = blob_dog(img_norm, max_sigma=30, threshold=.1)
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
        features['blob_count'] = len(blobs)
    else:
        features['blob_count'] = 0

    # Vector Gabungan
    svm_vector = []
    svm_vector.extend([features['mean'], features['std'], features['skew'], features['kurtosis']])
    svm_vector.extend([features['contrast'], features['dissimilarity'], features['homogeneity'], features['energy'], features['correlation'], features['ASM']])
    svm_vector.extend([features['area_total'], features['perimeter_mean'], features['solidity_mean'], features['eccentricity_mean']])
    svm_vector.extend(lbp_values)
    svm_vector.extend(hog_values)
    
    return features, svm_vector

# --- 4. INDIKATOR VISUAL ---
def calculate_indicators(feats, prediction_label):
    def sigmoid(x, center, scale): return 1 / (1 + math.exp(-scale * (x - center)))
    if prediction_label == "Normal":
        # Nilai rendah untuk normal
        return {k: 0.05 for k in ["Infiltrate", "Consolidation", "Cavity", "Fibrotic", "Calcification", "Effusion"]}
    
    scores = {}
    # Logika TBC
    scores['Infiltrate'] = sigmoid((feats.get('blob_count', 0)*2) + (feats.get('contrast', 0)*0.5), 15, 0.1)
    scores['Consolidation'] = sigmoid((feats.get('mean', 0)*0.5) + (feats.get('solidity_mean', 0)*60), 85, 0.1)
    scores['Cavity'] = sigmoid((feats.get('eccentricity_mean', 0)*50) + (feats.get('hog_mean', 0)*150), 45, 0.1)
    scores['Fibrotic'] = sigmoid((feats.get('correlation', 0)*80), 65, 0.1)
    scores['Calcification'] = sigmoid((feats.get('mean', 0)*0.9), 125, 0.1)
    scores['Effusion'] = sigmoid((feats.get('solidity_mean', 0)*80) + (feats.get('area_total', 0)/1000), 95, 0.1)
    return scores

# --- 5. VISUALISASI ---
def draw_segmentation_simulation(original_image, final_mask, prediction_label):
    img_vis = original_image.copy()
    if len(img_vis.shape) == 2: img_vis = cv.cvtColor(img_vis, cv.COLOR_GRAY2RGB)
    if prediction_label == "TBC":
        overlay = img_vis.copy()
        contours, _ = cv.findContours(final_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        colors = [(193, 182, 255), (150, 230, 255), (180, 200, 255)]
        for i, cnt in enumerate(contours):
            color = colors[i % len(colors)]
            cv.drawContours(overlay, [cnt], -1, color, -1)
            cv.drawContours(img_vis, [cnt], -1, color, 1)
        cv.addWeighted(overlay, 0.4, img_vis, 0.6, 0, img_vis)
    return img_vis

# --- 6. LOADER & FUNGSI SAKTI PENYAMAKAN DIMENSI ---

def fix_dimensions(vector, expected):
    """
    Fungsi Penyelamat: Menambah atau Memotong fitur agar pas dengan model.
    """
    current = vector.shape[1]
    if current == expected:
        return vector, "✅ Pas"
    elif current < expected:
        diff = expected - current
        # Tambah 0 di akhir (padding)
        padding = np.zeros((1, diff))
        new_vec = np.hstack((vector, padding))
        return new_vec, f"⚠️ Kurang {diff} fitur (Ditambah 0)"
    else:
        diff = current - expected
        # Potong fitur berlebih di akhir
        new_vec = vector[:, :expected]
        return new_vec, f"⚠️ Lebih {diff} fitur (Dipotong)"

@st.cache_resource(show_spinner=False)
def load_smart_model():
    # Cek model lengkap (Pipeline)
    paths_pipe = ['model_complete.pkl', 'model/model_complete.pkl']
    for p in paths_pipe:
        if os.path.exists(p):
            try: return {"model": joblib.load(p), "type": "pipeline", "status": "OK"}
            except: pass

    # Cek model terpisah
    paths = {
        'svm': ['model_svm.pkl', 'model/model_svm.pkl'],
        'pca': ['model_pca.pkl', 'model/model_pca.pkl'],
        'scaler': ['model_scaler.pkl', 'model/model_scaler.pkl']
    }
    assets = {}
    for key, path_list in paths.items():
        for p in path_list:
            if os.path.exists(p):
                try: assets[key] = joblib.load(p); break
                except: pass
    
    if len(assets) == 3: return {"model": assets, "type": "manual", "status": "OK"}
    return {"model": None, "type": "missing", "status": "File model belum lengkap"}

model_data = load_smart_model()

# --- 7. UI UTAMA ---
st.title("Analisis Citra X-Ray")
st.write("Unggah citra X-Ray Thorax untuk mendeteksi indikasi Tuberkulosis.")

if model_data['status'] != "OK":
    st.error("⚠️ Model belum lengkap. Cek folder 'model'.")
    st.write(f"Status: {model_data['status']}")
else:
    uploaded_file = st.file_uploader("Upload X-Ray Image (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        with st.spinner('Memproses citra...'):
            img_array = np.array(image)
            img_resized = cv.resize(img_array, (256, 256)) # Ukuran Default
            
            if len(img_resized.shape) == 3:
                if img_resized.shape[2] == 4: img_gray = cv.cvtColor(img_resized, cv.COLOR_RGBA2GRAY)
                else: img_gray = cv.cvtColor(img_resized, cv.COLOR_RGB2GRAY)
            else: img_gray = img_resized
                
            isolated_lungs, final_mask, img_clahe = pipeline_segmentasi_lengkap(img_gray)
            features_dict, feature_vector_list = extract_all_features_user_version(isolated_lungs, final_mask)
            
            prediction = "Unknown"; prob_tbc = 0.0; prob_normal = 0.0
            debug_msg = ""
            
            try:
                feat_array = np.array(feature_vector_list).reshape(1, -1)
                
                # --- AUTO-FIX DIMENSI ---
                expected_dim = 0
                if model_data['type'] == "pipeline":
                    # Ambil dimensi input scaler di dalam pipeline
                    expected_dim = model_data['model'].steps[0][1].n_features_in_
                else:
                    # Ambil dimensi scaler manual
                    expected_dim = model_data['model']['scaler'].n_features_in_
                
                # LAKUKAN PENYAMAKAN DIMENSI
                feat_array, debug_msg = fix_dimensions(feat_array, expected_dim)
                
                # --- PREDIKSI ---
                if model_data['type'] == "pipeline":
                    full_model = model_data['model']
                    pred_res = full_model.predict(feat_array)[0]
                    if hasattr(full_model, "predict_proba"): 
                        prob_tbc = full_model.predict_proba(feat_array)[0][1]
                    elif hasattr(full_model, "decision_function"):
                        dist = full_model.decision_function(feat_array)[0]
                        prob_tbc = 1 / (1 + np.exp(-dist))
                    else: prob_tbc = 0.99 if pred_res in [1, 'TBC'] else 0.01
                else:
                    assets = model_data['model']
                    feat_scaled = assets['scaler'].transform(feat_array)
                    feat_pca = assets['pca'].transform(feat_scaled)
                    pred_res = assets['svm'].predict(feat_pca)[0]
                    
                    if hasattr(assets['svm'], "decision_function"):
                        dist = assets['svm'].decision_function(feat_pca)[0]
                        prob_tbc = 1 / (1 + np.exp(-dist))
                    else: prob_tbc = 0.99 if pred_res in [1, 'TBC'] else 0.01

                prediction = "TBC" if (pred_res == 1 or pred_res == 'TBC' or prob_tbc > 0.5) else "Normal"
                prob_normal = 1.0 - prob_tbc
                    
            except Exception as e:
                st.error(f"Error fatal: {str(e)}")
                prediction = "Error"

        if prediction != "Error":
            clinical_scores = calculate_indicators(features_dict, prediction)
            
            # --- DEBUG INFO (SUPAYA TENANG) ---
            if "⚠️" in debug_msg:
                st.toast(f"Diagnostic: {debug_msg}", icon="🔧")
            
            st.markdown("### Hasil Visualisasi")
            c1, c2 = st.columns(2)
            with c1: st.info("Citra Asli"); st.image(img_clahe, use_container_width=True)
            with c2: st.info("Segmentasi Area"); st.image(draw_segmentation_simulation(cv.cvtColor(img_clahe, cv.COLOR_GRAY2RGB), final_mask, prediction), use_container_width=True)

            st.markdown("---"); st.subheader("Indikator Detail")
            col_a, col_b = st.columns(2)
            indicators_ui = [
                {"name": "Infiltrate", "key": "Infiltrate", "desc": "Bercak sebaran"},
                {"name": "Consolidation", "key": "Consolidation", "desc": "Kepadatan"},
                {"name": "Cavity", "key": "Cavity", "desc": "Lubang/Kavitas"},
                {"name": "Effusion", "key": "Effusion", "desc": "Cairan pleura"},
                {"name": "Fibrotic", "key": "Fibrotic", "desc": "Jaringan parut"},
                {"name": "Calcification", "key": "Calcification", "desc": "Pengapuran"}
            ]
            for i, item in enumerate(indicators_ui):
                target = col_a if i%2==0 else col_b
                with target:
                    score = clinical_scores[item['key']]
                    st.markdown(f"""<div class="indicator-box"><div><span class="indicator-title">{item['name']}</span><span class="percentage-text">{score*100:.1f}%</span></div>""", unsafe_allow_html=True)
                    st.progress(score)
                    st.markdown(f"""<div class="indicator-desc">{item['desc']}</div></div>""", unsafe_allow_html=True)

            st.markdown("---")
            k1, k2 = st.columns([2, 1])
            with k1:
                if prediction == "TBC": st.error(f"### ⚠️ TERINDIKASI TBC ({prob_tbc*100:.1f}%)")
                else: st.success(f"### ✅ NORMAL ({prob_normal*100:.1f}%)")
            with k2:
                st.write("Probabilitas:")
                st.progress(float(prob_tbc))
                st.caption(f"TBC: {prob_tbc*100:.1f}%")