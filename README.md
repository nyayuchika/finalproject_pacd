# Klasifikasi Penyakit Tuberkulosis pada Citra X-Ray Thorax

Proyek ini bertujuan untuk merancang sistem klasifikasi penyakit Tuberkulosis (TBC) berbasis citra rontgen dada (*Thorax X-Ray*). Sistem ini dikembangkan menggunakan **metode pengolahan citra klasik (non-deep learning)** yang mengutamakan efisiensi komputasi.

🔗 **Link Demo Aplikasi:** [https://kelompok1-pacd-finalproject.streamlit.app/](https://kelompok1-pacd-finalproject.streamlit.app/)

## 👥 Tim Penyusun (Kelompok 1 PACD)
* **Nyayu Chika Marselina** (25/568182/PPA/07148) -> nyayuchika
* **Yullase Pratiwi** (24/550766/PPA/06955) -> yullasepratiwi
* **Fawwaz Rif’at Revista** (25/565782/PPA/07130) -> Fawwaz1233
* **Galih Prabasidi** (25/563010/PPA/07092) -> UsagiUG

---

## 📖 Latar Belakang
Tuberkulosis (TBC) masih menjadi salah satu penyakit menular paling mematikan di dunia dengan estimasi 10 juta kasus baru per tahun menurut WHO. Diagnosis awal umumnya melalui foto rontgen dada, namun interpretasi manual memiliki tantangan:
* Variabilitas antar radiolog yang memengaruhi konsistensi diagnosis.
* Kualitas citra rontgen yang rendah memperbesar peluang salah diagnosis.

Penelitian ini menawarkan solusi otomasi analisis citra rontgen untuk meningkatkan kualitas citra dan akurasi diagnosis.

## 🎯 Tujuan Penelitian
1.  **Segmentasi Paru:** Memperoleh *mask* paru yang bersih dan terpisah dari latar belakang menggunakan metode *thresholding*.
2.  **Ekstraksi Fitur:** Menggunakan metode GLCM, HOG, dan LBP.
3.  **Klasifikasi:** Menerapkan algoritma *Support Vector Machine* (SVM) untuk membedakan citra TBC dan Normal.
4.  **Evaluasi:** Mengukur performa sistem menggunakan metrik *akurasi, precisisi, recall, f1-score,* dan IoU.

---

## 📂 Dataset
Dataset yang digunakan berjumlah total **704 citra** yang merupakan gabungan dari dua sumber publik yang diperoleh dari Kaggle dengan link https://www.kaggle.com/datasets/iamtapendu/chest-x-ray-lungs-segmentation
* **Sumber:** * Montgomery (USA): 141 images
    * Shenzhen (China): 563 images
* **Distribusi Kelas:**
    * Normal: 359 citra
    * TBC: 345 citra

---

## ⚙️ Metodologi
Alur pemrosesan data dilakukan melalui tahapan berikut:

### 1. Preprocessing
Bertujuan meningkatkan kualitas citra dan memperjelas area paru serta lesi.
* Resize Citra.
* **Gaussian Filter**.
* **CLAHE** (*Contrast Limited Adaptive Histogram Equalization*).

### 2. Segmentasi
Proses pemisahan area paru dari *background*.
* **Otsu Thresholding**.
* **Flood Fill**.
* **Morphological Closing**.

### 3. Ekstraksi Fitur
Menggabungkan tekstur, bentuk, dan deteksi tepi:
* **GLCM** (*Gray Level Co-occurrence Matrix*): Menganalisis hubungan statistik antar pixel (contrast, energy, correlation, dll) pada sudut 0, 45, 135, 180, 225, dan 315.
* **LBP** (*Local Binary Pattern*): Uniform method, Radius=2, Points=16.
* **HOG** (*Histogram of Oriented Gradients*): Orientations=9, Grid=16x16, Normalisasi 2x2.
* **Sobel**: Deteksi tepi vertikal dan horizontal.

### 4. Klasifikasi (Classifier)
* **Split Data:** 80% Training : 20% Testing.
* **Normalisasi:** Min-Max Scaler.
* **Reduksi Dimensi:** PCA (*Principal Component Analysis*) dengan n-components=0.50.
* **Model:** SVM dengan parameter:
    * `C`: 100
    * `gamma`: 0.001
    * `kernel`: 'rbf'.

---

## 📊 Hasil Evaluasi Penelitian

Berikut adalah rincian performa sistem berdasarkan metrik evaluasi yang digunakan:

| Kategori Evaluasi | Metrik | Nilai | Analisis |
| :--- | :--- | :--- | :--- |
| **Segmentasi** | **IoU (Intersection over Union)** | **75.5%** | Segmentasi berhasil memisahkan paru dari *background*. |
| **Klasifikasi** | **Akurasi Rata-rata** | **77%** | Model SVM dengan fitur PCA menunjukkan performa klasifikasi yang lebih baik dibanding yang tidak menerapkan PCA. |
| **Klasifikasi** | **Recall (Kelas TBC)** | **80%** | Menunjukkan kemampuan yang baik dalam mendeteksi kasus positif TBC. |
| **Klasifikasi** | **F1-Score** | **Seimbang** | Model tidak bias terhadap salah satu kelas dan memiliki kinerja konsisten. |

---

## 🚀 Saran Pengembangan
Untuk pengembangan lebih lanjut, disarankan untuk:
1.  **Optimasi Hyperparameter:** Melakukan tuning SVM yang lebih menyeluruh untuk meningkatkan akurasi.
2.  **Seleksi Fitur:** Menggunakan metode seleksi yang lebih spesifik untuk menyaring fitur relevan.
3.  **Reduksi Dimensi Lanjutan:** Mengeksplorasi teknik selain PCA untuk representasi fitur yang lebih optimal.
4.  **Penambahan Data:** Menambah jumlah atau variasi citra untuk meningkatkan *robustness* model.
5.  **Eksplorasi Model Lain:** Mencoba algoritma klasifikasi selain SVM.
