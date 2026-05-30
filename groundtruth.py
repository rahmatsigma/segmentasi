import os
import cv2
import numpy as np
from rembg import remove

PATH_DATASET_UTAMA = 'dataset/betta_fish'
PATH_GROUND_TRUTH = 'dataset/ground_truth'
KATEGORI_IKAN = ['halfmoon', 'double_tail', 'serit', 'plakat']

print("=== Memulai Pembuatan Ground Truth Otomatis dengan AI ===")

for kategori in KATEGORI_IKAN:
    folder_sumber = os.path.join(PATH_DATASET_UTAMA, kategori)
    folder_gt = os.path.join(PATH_GROUND_TRUTH, kategori)
    
    if not os.path.exists(folder_gt):
        os.makedirs(folder_gt)
        
    if not os.path.exists(folder_sumber):
        continue

    print(f"\nMemproses kategori: {kategori.upper()}")
    list_gambar = [f for f in os.listdir(folder_sumber) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for nama_file in list_gambar:
        path_sumber = os.path.join(folder_sumber, nama_file)
        path_simpan = os.path.join(folder_gt, nama_file)
        
        if os.path.exists(path_simpan):
            continue

        try:
            with open(path_sumber, 'rb') as i:
                input_data = i.read()
            
            output_data = remove(input_data)
            
            nparr = np.frombuffer(output_data, np.uint8)
            img_bg_removed = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
            if img_bg_removed.shape[2] == 4:
                alpha_channel = img_bg_removed[:, :, 3]
            else:
                print(f" -> {nama_file} tidak memiliki transparansi, dilewati.")
                continue
                
            _, mask_ground_truth = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
            
            cv2.imwrite(path_simpan, mask_ground_truth)
            print(f" -> Berhasil membuat Ground Truth: {nama_file}")
            
        except Exception as e:
            print(f" -> Gagal memproses {nama_file}: {e}")

print("\n=======================================================")
print(f"SELESAI! Silakan cek folder '{PATH_GROUND_TRUTH}'")
print("=======================================================")