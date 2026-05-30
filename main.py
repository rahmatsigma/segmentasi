import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

PATH_DATASET_UTAMA = 'dataset/betta_fish'
PATH_GROUND_TRUTH = 'dataset/ground_truth'
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


# 2. FUNGSI EVALUASI AKURASI SEGMENTASI (Sesuai Bab 3.6 Proposal)
def hitung_akurasi_segmentasi(mask_otomatis, path_gt):
    # Membaca citra ground truth dalam mode grayscale
    img_gt = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
    if img_gt is None:
        return None
    
    # Penyeragaman dimensi ground truth ke resolusi target 512x512
    mask_gt = cv2.resize(img_gt, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Binarisasi masker murni (nilai 0 atau 1) untuk perhitungan luas area
    _, mask_bin_otomatis = cv2.threshold(mask_otomatis, 0, 1, cv2.THRESH_BINARY)
    _, mask_bin_gt = cv2.threshold(mask_gt, 127, 1, cv2.THRESH_BINARY)
    
    # Rumus Proposal: (Σ nilai pixel hasil segmentasi cocok / Σ nilai pixel ground truth) * 100%
    piksel_irisan = np.sum(cv2.bitwise_and(mask_bin_otomatis, mask_bin_gt))
    total_piksel_gt = np.sum(mask_bin_gt)
    
    if total_piksel_gt == 0:
        return 0.0
        
    persentase_akurasi = (piksel_irisan / total_piksel_gt) * 100
    return persentase_akurasi


# 3. PIPELINE SEGMENTASI
def segmentasi_tunggal(image_path, path_gt=None):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, None, None, None
    
    # Resize citra ke ukuran standar proposal (512x512)
    img_512 = cv2.resize(img_bgr, (512, 512))
    
    # Konversi ke ruang warna HSI
    hsi_img = bgr_to_hsi(img_512)
    
    # [MODIFIKASI UTAMA]: Downscale untuk proses clustering (128x128) guna optimasi komputasi
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
    
    # [UPSCALE]: Kembalikan ukuran mask ke resolusi asli proposal (512x512)
    mask_512 = cv2.resize(mask_small, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Potong objek ikan menggunakan mask biner
    hasil_segmentasi = cv2.bitwise_and(img_512, img_512, mask=mask_512)
    
    # Jalankan evaluasi metrik apabila file ground truth tersedia
    skor_evaluasi = None
    if path_gt and os.path.exists(path_gt):
        skor_evaluasi = hitung_akurasi_segmentasi(mask_512, path_gt)
    
    return img_512, mask_512, hasil_segmentasi, skor_evaluasi


# 4. PROSES UTAMA
print("=== Memulai Proses Segmentasi Otomatis & Evaluasi ===")

if not os.path.exists(PATH_OUTPUT):
    os.makedirs(PATH_OUTPUT)

total_terproses = 0
akumulasi_skor_akurasi = []

for kategori in KATEGORI_IKAN:
    folder_sumber = os.path.join(PATH_DATASET_UTAMA, kategori)
    folder_tujuan = os.path.join(PATH_OUTPUT, kategori)
    folder_gt = os.path.join(PATH_GROUND_TRUTH, kategori)
    
    if not os.path.exists(folder_tujuan):
        os.makedirs(folder_tujuan)
        
    if not os.path.exists(folder_sumber):
        print(f"Peringatan: Folder {folder_sumber} tidak ditemukan. Skip.")
        continue
        
    print(f"\nMemproses kategori: {kategori.upper()}")
    
    list_gambar = [f for f in os.listdir(folder_sumber) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for nama_file in list_gambar:
        path_file_gambar = os.path.join(folder_sumber, nama_file)
        path_file_gt = os.path.join(folder_gt, nama_file)
        
        asli, mask, hasil, akurasi = segmentasi_tunggal(path_file_gambar, path_file_gt)
        
        if hasil is not None:
            path_simpan = os.path.join(folder_tujuan, f"segmented_{nama_file}")
            cv2.imwrite(path_simpan, hasil)
            total_terproses += 1
            
            # Tampilkan informasi evaluasi jika masker acuan manual ada
            if akurasi is not None:
                print(f" -> Berhasil memproses: {nama_file} | Skor Akurasi: {akurasi:.2f}%")
                akumulasi_skor_akurasi.append(akurasi)
            else:
                print(f" -> Berhasil memproses: {nama_file} | (Ground truth tidak ditemukan, skip evaluasi)")
        else:
            print(f" -> Gagal membaca: {nama_file}")

print("\n=======================================================")
print(f"PROSES SELESAI! Total {total_terproses} gambar berhasil di-segmentasi.")
if akumulasi_skor_akurasi:
    rata_rata_akurasi = np.mean(akumulasi_skor_akurasi)
    print(f"Rata-rata Akurasi Segmentasi Sistem Keseluruhan: {rata_rata_akurasi:.2f}%")
print(f"Silakan cek hasilnya di dalam folder: '{PATH_OUTPUT}/'")
print("=======================================================")