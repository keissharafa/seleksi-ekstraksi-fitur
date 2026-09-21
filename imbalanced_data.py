import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# 1. Load Dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 2. Langkah Pengecekan
if 'Attrition' not in df.columns:
    print("Error: Kolom 'Attrition' tidak ditemukan dalam dataset.")
    exit()

print("--- Menampilkan 5 Baris Pertama ---")
print(df.head())

print("\n--- Mengecek Data Kosong (Missing Values) ---")
missing = df.isna().sum()
print(missing)

# 3. memisah Fitur (X) dan Target (y)
X = df.drop(columns=['Attrition']).select_dtypes(exclude=['object'])
y = df['Attrition'].map({'Yes': 1, 'No': 0})

# 4. Fungsi Pembuat Grafik 
def buat_plot(data_y, judul, nama_file):
    plt.figure(figsize=(6,4))
    distribusi = pd.Series(data_y).value_counts()
    sns.barplot(x=distribusi.index, y=distribusi.values, palette='coolwarm')
    plt.xticks(ticks=[0,1], labels=['Bertahan (0)', 'Resign (1)'])
    plt.xlabel("Status")
    plt.ylabel("Jumlah")
    plt.title(judul)
    
    # Menyimpan gambar dengan kualitas rapi (tidak terpotong)
    plt.savefig(nama_file, bbox_inches='tight')
    print(f"[SUCCESS] Gambar disimpan sebagai file: {nama_file}")
    
    plt.show()

# --- Distribusi SEBELUM Diseimbangkan ---
print("\nDistribusi Sebelum Diproses:")
print(y.value_counts())
buat_plot(y, "Distribusi Asli Sebelum Diproses", "1_Grafik_Asli.png")

# --- 1. Penerapan ROS (Random Over Sampler) ---
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X, y)
print("\nDistribusi Sesudah ROS:")
print(pd.Series(y_ros).value_counts())
buat_plot(y_ros, "Distribusi Sesudah ROS", "2_Grafik_ROS.png")

# --- 2. Penerapan RUS (Random Under Sampler) ---
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)
print("\nDistribusi Sesudah RUS:")
print(pd.Series(y_rus).value_counts())
buat_plot(y_rus, "Distribusi Sesudah RUS", "3_Grafik_RUS.png")

# --- 3. Penerapan SMOTE ---
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
print("\nDistribusi Sesudah SMOTE:")
print(pd.Series(y_smote).value_counts())
buat_plot(y_smote, "Distribusi Sesudah SMOTE", "4_Grafik_SMOTE.png")

# --- 4. Penerapan SMOTE-ENN ---
smote_enn = SMOTEENN(random_state=42)
X_smenn, y_smenn = smote_enn.fit_resample(X, y)
print("\nDistribusi Sesudah SMOTE-ENN:")
print(pd.Series(y_smenn).value_counts())
buat_plot(y_smenn, "Distribusi Sesudah SMOTE-ENN", "5_Grafik_SMOTE_ENN.png")