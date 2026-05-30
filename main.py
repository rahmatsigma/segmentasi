import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

PATH_DATASET_UTAMA = 'dataset/betta_fish'
KATEGORI_IKAN = ['halfmoon', 'double_tail', 'serit', 'plakat']
PATH_OUTPUT = 'output_segmentasi'


# 1. KONVERSI BGR KE HSI
def bgr_to_hsi(img):
    # Normalisasi nilai piksel ke rentang [0, 1]
    bgr = np.float32(img) / 255.0
    b, g, r = cv2.split(bgr)

    # Menghitung Intensity (I)
    i = (r + g + b) / 3.0

    # Menghitung Saturation (S)
    minimum = np.minimum(np.minimum(r, g), b)
    s = 1 - (3 * minimum / (r + g + b + 1e-5)) 

    # Menghitung Hue (H)
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g)**2 + (r - b) * (g - b))
    theta = np.arccos(num / (den + 1e-5))
    
    # Kondisi penentuan nilai Hue dalam radian
    h = np.where(b <= g, theta, 2 * np.pi - theta)
    h = h / (2 * np.pi) 

    # Gabungkan kembali menjadi citra HSI
    hsi_img = cv2.merge((h, s, i))
    return hsi_img


# 2. PIPELINE SEGMENTASI
def segmentasi_tunggal(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, None, None
    
    # Resize citra ke ukuran standar proposal (512x512)
    img_512 = cv2.resize(img_bgr, (512, 512))
    
    # Konversi ke ruang warna HSI
    hsi_img = bgr_to_hsi(img_512)
    
    # Downscale untuk proses clustering (128x128)
    hsi_small = cv2.resize(hsi_img, (128, 128), interpolation=cv2.INTER_AREA)
    
    # Fitur warna: Hue dan Saturation
    fitur_warna = hsi_small[:, :, :2] 
    X = fitur_warna.reshape((-1, 2))
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.03, min_samples=30)
    labels = dbscan.fit_predict(X)
    
    # Label ke matriks 2D (128x128)
    labels_2d = labels.reshape((128, 128))
    
    # Deteksi Background (Cluster paling dominan/luas adalah background)
    unik, hitung = np.unique(labels_2d, return_counts=True)
    background_label = unik[np.argmax(hitung)]
    
    # Binary Mask (0 untuk background, 1 untuk objek ikan)
    mask_small = np.where(labels_2d == background_label, 0, 1).astype(np.uint8)
    
    # Kembalikan (Upscale) ukuran mask ke resolusi asli proposal (512x512)
    mask_512 = cv2.resize(mask_small, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Potong objek ikan menggunakan mask biner
    hasil_segmentasi = cv2.bitwise_and(img_512, img_512, mask=mask_512)
    
    return img_512, mask_512, hasil_segmentasi


# 3. PROSES UTAMA
print("=== Memulai Proses Segmentasi Otomatis ===")

if not os.path.exists(PATH_OUTPUT):
    os.makedirs(PATH_OUTPUT)

total_terproses = 0

for kategori in KATEGORI_IKAN:
    folder_sumber = os.path.join(PATH_DATASET_UTAMA, kategori)
    folder_tujuan = os.path.join(PATH_OUTPUT, kategori)
    
    if not os.path.exists(folder_tujuan):
        os.makedirs(folder_tujuan)
        
    if not os.path.exists(folder_sumber):
        print(f"Peringatan: Folder {folder_sumber} tidak ditemukan. Skip.")
        continue
        
    print(f"\nMemproses kategori: {kategori.upper()}")
    
    list_gambar = [f for f in os.listdir(folder_sumber) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for nama_file in list_gambar:
        path_file_gambar = os.path.join(folder_sumber, nama_file)
        
        asli, mask, hasil = segmentasi_tunggal(path_file_gambar)
        
        if hasil is not None:
            path_simpan = os.path.join(folder_tujuan, f"segmented_{nama_file}")
            cv2.imwrite(path_simpan, hasil)
            total_terproses += 1
            print(f" -> Berhasil memproses: {nama_file}")
        else:
            print(f" -> Gagal membaca: {nama_file}")

print("\n=======================================================")
print(f"PROSES SELESAI! Total {total_terproses} gambar berhasil di-segmentasi.")
print(f"Silakan cek hasilnya di dalam folder: '{PATH_OUTPUT}/'")
print("=======================================================")