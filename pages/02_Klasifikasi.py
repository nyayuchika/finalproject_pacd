import streamlit as st
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from scipy import ndimage
import joblib

# --- 1. KONFIGURASI HALAMAN & STYLING ---
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
    
    [data-testid="stFileUploader"] { 
        padding: 20px; border: 2px dashed #008CFF; border-radius: 10px; background-color: #F0F8FF; 
    }
    h1, h2, h3 { color: #008CFF; }
</style>
""", unsafe_allow_html=True)

st.title("🫁 Deteksi TBC Paru-paru")
st.write("Unggah citra X-Ray paru-paru untuk melakukan prediksi Normal atau TBC.")

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("model/77_percent/model_scaler.pkl")
        pca = joblib.load("model/77_percent/model_pca.pkl")
        svm = joblib.load("model/77_percent/model_svm.pkl")
        return scaler, pca, svm
    except FileNotFoundError:
        st.error("File model (.pkl) tidak ditemukan!")
        return None, None, None

# --- 3. FUNGSI PREPROCESSING ---
def process_image(img_array):
    # Resize & Preprocessing dasar
    img = cv2.resize(img_array, (256, 256), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gauss)

    # Segmentasi (Otsu + Morph + Connected Components)
    _, otsu_bin = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    padded = cv2.copyMakeBorder(otsu_bin, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, mask, (0, 0), 0)
    flooded = padded[1:-1, 1:-1]
    
    kernel = np.ones((11, 11), np.uint8)
    closed = cv2.morphologyEx(flooded, cv2.MORPH_CLOSE, kernel)
    filled = ndimage.binary_fill_holes(closed).astype(np.uint8) * 255

    num_labels, labels = cv2.connectedComponents(filled)
    lung_mask = np.zeros_like(filled) 
    if num_labels > 1:
        areas = [(labels == i).sum() for i in range(1, num_labels)]
        k = min(2, len(areas))
        largest = np.argsort(areas)[-k:] + 1
        lung_mask = np.isin(labels, largest).astype(np.uint8) * 255

    # Visualisasi Bounding Box
    img_bbox = img.copy()
    contours, _ = cv2.findContours(lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Masking final
    isolated_lung = cv2.bitwise_and(clahe_img, clahe_img, mask=lung_mask)
    masked_img = isolated_lung.astype(np.uint8)

    # --- FEATURE EXTRACTION ---
    
    # 1. GLCM
    angles = [np.pi/4, np.pi/2, 3*np.pi/4, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    glcm = graycomatrix(masked_img, distances=[1], angles=angles, levels=256, symmetric=True, normed=True)
    # Urutan props: contrast, dissimilarity, homogeneity, energy, correlation, ASM
    glcm_props = [graycoprops(glcm, prop)[0, 0] for prop in 
                  ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]

    # 2. LBP
    lbp = local_binary_pattern(masked_img, P=12, R=2, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # 3. HOG
    hog_features = hog(masked_img, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), visualize=False)

    # 4. Sobel
    sobel_x = cv2.Sobel(masked_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(masked_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mean = np.mean(sobel_mag)
    sobel_std = np.std(sobel_mag)

    # Gabungkan fitur untuk Model SVM
    feature_vector = np.hstack([glcm_props, lbp_hist, hog_features, [sobel_mean, sobel_std]])
    feature_vector = feature_vector.reshape(1, -1)

    return feature_vector, img, img_bbox

# --- 4. UI UTAMA ---
uploaded_file = st.file_uploader("Pilih gambar X-Ray...", type=["jpg", "jpeg", "png"])
# --- NAVIGASI: TOMBOL KEMBALI KE HOME (DI HALAMAN UTAMA) ---
if st.button("Kembali ke Home"):
    try:
        st.switch_page("01_Home.py")
    except Exception as e:
        st.error(f"Gagal pindah halaman. Pastikan file '01_Home.py' ada di folder yang sama atau folder 'pages/'. Error: {e}")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, 1)

    if img_cv is not None:
        with st.spinner('Menganalisis tekstur paru-paru...'):
            scaler, pca, svm = load_models()

            if scaler is not None:
                # Ambil features dan gambar (tanpa indikator detail)
                features, original_img, bbox_img = process_image(img_cv)

                # Prediksi
                scaled_features = scaler.transform(features)
                reduced_features = pca.transform(scaled_features)
                prediction = svm.predict(reduced_features)[0]
                
                label = "NORMAL" if prediction == 0 else "TBC"

                # --- TAMPILAN GAMBAR ---
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h3 style='text-align: center;'>Citra Asli</h3>", unsafe_allow_html=True)
                    st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with col2:
                    st.markdown("<h3 style='text-align: center;'>Area Analisis</h3>", unsafe_allow_html=True)
                    st.image(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB), caption="ROI Paru Terdeteksi", use_container_width=True)

                st.divider()

                # --- TAMPILAN DIAGNOSA UTAMA ---
                if label == "NORMAL":
                    st.markdown(f"""
                    <div class="indicator-box" style="border-left: 5px solid #28a745;">
                        <div class="indicator-title">HASIL DIAGNOSA: ✅ NORMAL</div>
                        <div class="indicator-desc">Paru-paru terdeteksi sehat. Struktur tekstur dan kepadatan dalam batas wajar.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="indicator-box" style="border-left: 5px solid #dc3545;">
                        <div class="indicator-title">HASIL DIAGNOSA: ⚠️ TBC</div>
                        <div class="indicator-desc">Terdeteksi anomali tekstur yang mengindikasikan Tuberculosis.</div>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.error("Format gambar tidak valid.")