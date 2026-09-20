import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind
import warnings
warnings.filterwarnings('ignore')

print("=== PROGRAM SELEKSI FITUR (IBM HR ATTRITION) ===\n")

# 1. MEMUAT DATASET
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
target = 'Attrition'

# 2. MEMISAHKAN FITUR KATEGORIKAL DAN NUMERIKAL
kolom_kategorikal = df.select_dtypes(include=['object']).columns.drop(target).tolist() #kolom berupa string
kolom_numerik = df.select_dtypes(exclude=['object']).columns.tolist()

print(f"Total Fitur Kategorikal : {len(kolom_kategorikal)}")
print(f"Total Fitur Numerikal   : {len(kolom_numerik)}\n")

# =====================================================================
# TAHAP 1: SELEKSI FITUR KATEGORIKAL MENGGUNAKAN CHI-SQUARE
# =====================================================================
print("=== HASIL UJI CHI-SQUARE (FITUR KATEGORIKAL) ===")
fitur_kategorik_terpilih = []

for col in kolom_kategorikal:
    # Membuat tabel kontingensi
    contingency_table = pd.crosstab(df[col], df[target])
    
    # Melakukan uji chi-square
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Syarat mutlak: P-Value harus < 0.05 (Signifikan)
    if p < 0.05:
        fitur_kategorik_terpilih.append(col)
        print(f"[DIPILIH] {col.ljust(20)} | P-Value: {p:.5f} (Signifikan)")
    else:
        print(f"[DIBUANG] {col.ljust(20)} | P-Value: {p:.5f}")

# =====================================================================
# TAHAP 2: SELEKSI FITUR NUMERIKAL MENGGUNAKAN T-TEST
# =====================================================================
print("\n=== HASIL UJI T-TEST (FITUR NUMERIKAL) ===")
fitur_numerik_terpilih = []

# Memisahkan data berdasarkan target (Karyawan Resign vs Tidak Resign)
df_yes = df[df[target] == 'Yes']
df_no = df[df[target] == 'No']

for col in kolom_numerik:
    # Melakukan uji T-Test Independent
    stat, p = ttest_ind(df_yes[col], df_no[col], nan_policy='omit')
    
    # Syarat mutlak: P-Value harus < 0.05 (Signifikan)
    # Tambahan: menolak nilai p yang NaN (biasanya karena kolom isinya sama semua)
    if pd.notna(p) and p < 0.05:
        fitur_numerik_terpilih.append(col)
        print(f"[DIPILIH] {col.ljust(20)} | P-Value: {p:.5f} (Signifikan)")
    else:
        print(f"[DIBUANG] {col.ljust(20)} | P-Value: {p:.5f}")

# =====================================================================
# TAHAP 3: MENYIMPAN KE DALAM 2 FILE CSV (SESUAI INSTRUKSI DOSEN)
# =====================================================================
# File 1: Khusus Kategorikal
df_kategorik_final = df[fitur_kategorik_terpilih + [target]]
df_kategorik_final.to_csv("Fitur_Kategorikal_Terpilih.csv", index=False)

# File 2: Khusus Numerikal
df_numerik_final = df[fitur_numerik_terpilih + [target]]
df_numerik_final.to_csv("Fitur_Numerikal_Terpilih.csv", index=False)

print("\n=== STATUS AKHIR ===")
print("[FILE DISIMPAN] 'Fitur_Kategorikal_Terpilih.csv' berhasil dibuat.")
print("[FILE DISIMPAN] 'Fitur_Numerikal_Terpilih.csv' berhasil dibuat.")
print("Proses Seleksi Fitur 100% Selesai!")

# =====================================================================
# TAHAP 4: VISUALISASI HASIL SELEKSI (BONUS GRAFIK)
# =====================================================================
print("\n=== TAHAP 4: MEMBUAT GRAFIK VISUALISASI ===")

# Mengambil angka dinamis langsung dari variabel yang sudah dihitung 
kategori_labels = ['Fitur Kategorikal', 'Fitur Numerikal', 'Total Fitur']
sebelum = [len(kolom_kategorikal), len(kolom_numerik), len(kolom_kategorikal) + len(kolom_numerik)]
sesudah = [len(fitur_kategorik_terpilih), len(fitur_numerik_terpilih), len(fitur_kategorik_terpilih) + len(fitur_numerik_terpilih)]

x = np.arange(len(kategori_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, sebelum, width, label='Sebelum Seleksi', color='#FF9999', edgecolor='black')
rects2 = ax.bar(x + width/2, sesudah, width, label='Sesudah Seleksi', color='#66B2FF', edgecolor='black')

ax.set_ylabel('Jumlah Kolom/Fitur', fontsize=12)
ax.set_title('Perbandingan Dimensi Dataset (Before vs After Seleksi Fitur)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(kategori_labels, fontsize=11)
ax.legend()

ax.bar_label(rects1, padding=3, fontsize=11, fontweight='bold')
ax.bar_label(rects2, padding=3, fontsize=11, fontweight='bold')

fig.tight_layout()

# Menyimpan grafik sebagai file PNG secara otomatis
plt.savefig('Visualisasi_Seleksi_Fitur.png')
print("[FILE DISIMPAN] Grafik 'Visualisasi_Seleksi_Fitur.png' berhasil dibuat!")

# Memunculkan popup grafik di layar
plt.show()