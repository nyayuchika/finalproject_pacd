# Klasifikasi Penyakit Tuberkulosis pada Citra X-Ray Thorax

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kelompok1-pacd-finalproject.streamlit.app/)

Proyek ini bertujuan untuk merancang sistem klasifikasi penyakit Tuberkulosis (TBC) berbasis citra rontgen dada (*Thorax X-Ray*). [cite_start]Sistem ini dikembangkan menggunakan **metode pengolahan citra klasik (non-deep learning)** yang mengutamakan efisiensi komputasi tanpa memerlukan dataset masif[cite: 18, 28, 33].

🔗 **Link Demo Aplikasi:** [https://kelompok1-pacd-finalproject.streamlit.app/](https://kelompok1-pacd-finalproject.streamlit.app/)

## 👥 Tim Penyusun (Kelompok 1 PACD)
* [cite_start]**Nyayu Chika Marselina** (25/568182/PPA/07148) -> nyayuchika [cite: 2]
* [cite_start]**Yullase Pratiwi** (24/550766/PPA/06955) -> yullasepratiwi [cite: 3]
* [cite_start]**Fawwaz Rif’at Revista** (25/565782/PPA/07130) -> Fawwaz1233 [cite: 4]
* [cite_start]**Galih Prabasidi** (25/563010/PPA/07092) -> UsagiUG [cite: 5]

---

## 📖 Latar Belakang
[cite_start]Tuberkulosis (TBC) masih menjadi salah satu penyakit menular paling mematikan di dunia dengan estimasi 10 juta kasus baru per tahun menurut WHO[cite: 9, 10]. Diagnosis awal umumnya melalui foto rontgen dada, namun interpretasi manual memiliki tantangan:
* [cite_start]Variabilitas antar radiolog yang memengaruhi konsistensi diagnosis[cite: 12].
* [cite_start]Kualitas citra rontgen yang rendah memperbesar peluang salah diagnosis[cite: 13].

[cite_start]Penelitian ini menawarkan solusi otomasi analisis citra rontgen untuk meningkatkan kualitas citra dan akurasi diagnosis[cite: 18].

## 🎯 Tujuan Penelitian
1.  [cite_start]**Segmentasi Paru:** Memperoleh *mask* paru yang bersih dan terpisah dari latar belakang menggunakan metode *thresholding*[cite: 34, 45].
2.  [cite_start]**Ekstraksi Fitur:** Menggunakan metode GLCM, HOG, dan LBP[cite: 46].
3.  [cite_start]**Klasifikasi:** Menerapkan algoritma *Support Vector Machine* (SVM) untuk membedakan citra TBC dan Normal[cite: 35, 39].
4.  [cite_start]**Evaluasi:** Mengukur performa sistem menggunakan metrik *accuracy, precision, recall, f1-score,* dan IoU[cite: 36, 40].

---

## 📂 Dataset
[cite_start]Dataset yang digunakan berjumlah total **704 citra** yang merupakan gabungan dari dua sumber publik[cite: 55, 56, 60, 61]:
* **Sumber:** * Montgomery (USA): 141 images
    * Shenzhen (China): 563 images
* **Distribusi Kelas:**
    * [cite_start]Normal: 359 citra [cite: 51, 52]
    * [cite_start]TBC: 345 citra [cite: 53, 54]

---

## ⚙️ Metodologi
Alur pemrosesan data dilakukan melalui tahapan berikut:

### 1. Preprocessing
[cite_start]Bertujuan meningkatkan kualitas citra dan memperjelas area paru serta lesi[cite: 112].
* [cite_start]Resize Citra[cite: 64].
* [cite_start]**Gaussian Filter**[cite: 65].
* [cite_start]**CLAHE** (*Contrast Limited Adaptive Histogram Equalization*)[cite: 67].

### 2. Segmentasi
Proses pemisahan area paru dari *background*.
* [cite_start]**Otsu Thresholding**[cite: 69].
* [cite_start]**Flood Fill**[cite: 70].
* [cite_start]**Morphological Closing**[cite: 71].

### 3. Ekstraksi Fitur
Menggabungkan tekstur, bentuk, dan deteksi tepi:
* [cite_start]**GLCM** (*Gray Level Co-occurrence Matrix*): Menganalisis hubungan statistik antar pixel (contrast, energy, correlation, dll) pada sudut 0, 45, 135, 180, 225, dan 315 [cite: 74-77].
* [cite_start]**LBP** (*Local Binary Pattern*): Uniform method, Radius=2, Points=16 [cite: 80-83].
* [cite_start]**HOG** (*Histogram of Oriented Gradients*): Orientations=9, Grid=16x16, Normalisasi 2x2 [cite: 84-89].
* [cite_start]**Sobel**: Deteksi tepi vertikal dan horizontal[cite: 79, 90, 91].

### 4. Klasifikasi (Classifier)
* [cite_start]**Split Data:** 80% Training : 20% Testing[cite: 96].
* [cite_start]**Normalisasi:** Min-Max Scaler[cite: 98].
* [cite_start]**Reduksi Dimensi:** PCA (*Principal Component Analysis*) dengan n-components=0.50[cite: 99, 100].
* **Model:** SVM (*Support Vector Machine*) dengan parameter:
    * `C`: 100
    * `gamma`: 0.001
    * [cite_start]`kernel`: 'rbf'[cite: 101, 102].

---

## 📊 Hasil Evaluasi Penelitian

Berikut adalah rincian performa sistem berdasarkan metrik evaluasi yang digunakan:

| Kategori Evaluasi | Metrik | Nilai | Analisis |
| :--- | :--- | :--- | :--- |
| **Segmentasi** | **IoU (Intersection over Union)** | **75.5%** | [cite_start]Segmentasi berhasil memisahkan paru dari *background* dengan baik[cite: 113]. |
| **Klasifikasi** | **Akurasi Rata-rata** | **77%** | [cite_start]Model SVM dengan fitur PCA menunjukkan performa klasifikasi yang kompetitif[cite: 115]. |
| **Klasifikasi** | **Recall (Kelas TBC)** | **80%** | [cite_start]Menunjukkan kemampuan yang baik dalam mendeteksi kasus positif TBC[cite: 114]. |
| **Klasifikasi** | **F1-Score** | **Seimbang** | [cite_start]Model tidak bias terhadap salah satu kelas dan memiliki kinerja konsisten[cite: 117]. |

> [cite_start]**Catatan:** Kombinasi fitur GLCM, HOG, dan LBP terbukti memberikan representasi karakteristik tekstur dan struktur yang informatif untuk proses klasifikasi ini[cite: 116].

---

## 🚀 Saran Pengembangan
Untuk pengembangan lebih lanjut, disarankan untuk:
1.  [cite_start]**Optimasi Hyperparameter:** Melakukan tuning SVM yang lebih menyeluruh untuk meningkatkan akurasi[cite: 123].
2.  [cite_start]**Seleksi Fitur:** Menggunakan metode seleksi yang lebih spesifik untuk menyaring fitur relevan[cite: 124].
3.  [cite_start]**Reduksi Dimensi Lanjutan:** Mengeksplorasi teknik selain PCA untuk representasi fitur yang lebih optimal[cite: 126].
4.  [cite_start]**Penambahan Data:** Menambah jumlah atau variasi citra untuk meningkatkan *robustness* model[cite: 128].
5.  [cite_start]**Eksplorasi Model Lain:** Mencoba algoritma klasifikasi selain SVM[cite: 130].

---
*Dibuat berdasarkan Laporan Kelompok 1 PACD - Universitas Gadjah Mada*.
