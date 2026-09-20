# IBM HR Analytics: Feature Selection & Principal Component Analysis (PCA)

Repository ini berisi rangkaian proses *Machine Learning* prapemrosesan data, mulai dari **Seleksi Fitur** hingga **Ekstraksi Fitur (Dimensionality Reduction menggunakan PCA)** pada dataset **IBM HR Employee Attrition**.

---

## 📂 Struktur Proyek & Penjelasan Folder
* `seleksi_fitur.py` : Kode program untuk menyaring fitur numerikal dan kategorikal yang relevan.
* `ekstraksi_fitur.py` : Kode program utama untuk standarisasi Z-Score, reduksi dimensi PCA 2 Komponen, dan pencarian Komponen Optimal (Threshold 80%).
* `Ekstraksi_2_Komponen.csv` : Hasil proyeksi data ke dalam 2 Komponen Utama (PC1 & PC2) beserta target *Attrition*.
* `Ekstraksi_Komponen_Optimal.csv` : Hasil ekstraksi dengan 14 Komponen Utama optimal (merangkum >81% varians data).
* `Scatter_Plot_PCA.png` : Visualisasi sebaran data 2 Dimensi berdasarkan status *Attrition*.

---

## 📊 Ringkasan Metodologi
1. **Prapemrosesan & Standarisasi:** 
   * Memfilter 26 fitur numerikal murni dari total dataset.
   * Melakukan normalisasi menggunakan `StandardScaler` (Z-Score) agar skala data seragam.
2. **Ekstraksi Fitur Statis (PCA 2 Komponen):**
   * Digunakan untuk kebutuhan visualisasi spasial 2 Dimensi.
   * Menghasilkan *Total Variance Explained* sebesar **27.05%** (menunjukkan adanya *overlapping* kelas secara visual).
3. **Pencarian Komponen Optimal (Threshold 80%):**
   * Menggunakan pendekatan kumulatif varians otomatis.
   * Berhasil mereduksi dimensi dari 26 fitur menjadi **14 Komponen Utama** dengan *Total Variance Explained* sebesar **81.36%**.

---

## 🚀 Cara Menjalankan Program
Pastikan pustaka Python berikut sudah terinstal di lingkungan komputermu (`pandas`, `numpy`, `matplotlib`, `scikit-learn`), lalu jalankan skrip melalui terminal VSCode:

```bash
python ekstraksi_fitur.py
