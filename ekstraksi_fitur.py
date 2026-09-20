import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load Data Mentah IBM
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 2. Pisahkan Target & Ambil Fitur Numerik Saja (Syarat PCA)
target = df["Attrition"]
X = df.select_dtypes(exclude=['object'])

# 3. Standarisasi Data / normalisasi dg z-score
scaler = StandardScaler() # manggil alat z-score
X_scaled = scaler.fit_transform(X) #fit = menghitung rata-rata dan std pd setiap kolom numerikal (x), transform = mengubah data ke z-score

# A: EKSTRAKSI 2 KOMPONEN (n komponen bebas)
pca_2 = PCA(n_components=2)
# Fungsi fit_transform:
# 1. 'fit'       : menghitung matriks kovarian dan nilai eigen untuk mencari pola varians (informasi) terbesar dari data X_scaled.
# 2. 'transform' : mengeksekusi ekstraksi dengan memeras data asli menjadi hanya 2 kolom koordinat baru (PC1 dan PC2).
X_pca_2 = pca_2.fit_transform(X_scaled) 

# ngubah hasil ekstraksi PCA menjadi DataFrame agar bisa disimpan ke CSV
df_pca_2 = pd.DataFrame(X_pca_2, columns=["PC1", "PC2"]) #angka mentah diubah ke table
df_pca_2["Outcome"] = target #kolom target dimasukkan jg ke table
df_pca_2.to_csv("Ekstraksi_2_Komponen.csv", index=False) 

print("=== PCA 2 KOMPONEN ===")
print("Explained Variance Ratio:", pca_2.explained_variance_ratio_)
print("Total Variance Explained:", sum(pca_2.explained_variance_ratio_))

# B: EKSTRAKSI KOMPONEN OPTIMAL (n komponen optimal)
# Parameter 0.80 otomatis mencari jumlah n yang mempertahankan 80% informasi
pca_opt = PCA(n_components=0.80) 
X_pca_opt = pca_opt.fit_transform(X_scaled)

kolom_opt = [f"PC{i+1}" for i in range(X_pca_opt.shape[1])] #buat daftar nama kolom otomatis
df_pca_opt = pd.DataFrame(X_pca_opt, columns=kolom_opt) 
df_pca_opt["Outcome"] = target
df_pca_opt.to_csv("Ekstraksi_Komponen_Optimal.csv", index=False)

print(f"\n=== PCA OPTIMAL ({X_pca_opt.shape[1]} KOMPONEN) ===")
print("Explained Variance Ratio:\n", pca_opt.explained_variance_ratio_)
print("Total Variance Explained:", sum(pca_opt.explained_variance_ratio_))

# --- VISUALISASI HASIL PCA (2 KOMPONEN) ---
# Mengatur palet warna
warna = df_pca_2["Outcome"].map({'Yes': 1, 'No': 0}) 

plt.figure(figsize=(8, 6))
scatter = plt.scatter(df_pca_2["PC1"], df_pca_2["PC2"], c=warna, cmap="coolwarm", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: IBM HR Dataset")
plt.colorbar(scatter, label="Attrition (1=Yes, 0=No)")
plt.savefig("Scatter_Plot_PCA.png")
plt.show()