import os

# Resim dosyalarının bulunduğu klasör yolu
folder_path = os.path.join(os.path.dirname(__file__), "spot_data/spot_images")
#baska bir kullanim
# Desteklenen resim dosyası uzantıları
image_extensions = [".jpg", ".jpeg", ".png"]

# Kullanıcıdan kaç fotoğrafın listeleneceğini al
max_count = int(input("Kaç fotoğrafin yolunu görmek istiyorsunuz? "))

# Kayıt dosyasının adı   
output_file = "spot_texting.txt"

# Sayaç başlat
counter = 0

# Yazma modunda dosyayı aç
with open(output_file, "w") as file:
    # Klasördeki resimlerin yollarını listele
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if counter >= max_count:  # Belirtilen sayıya ulaşıldığında dur
                break
            if any(file_name.lower().endswith(ext) for ext in image_extensions):
                # Tam yol yerine göreceli yolu al
                relative_path = os.path.relpath(os.path.join(root, file_name), os.path.dirname(os.path.dirname(folder_path)))
                print(relative_path)  # Konsola yazdır
                file.write(relative_path + "\n")  # Dosyaya yaz
                os.remove(os.path.join(root, file_name))  # Dosyayı sil
                counter += 1
        if counter >= max_count:
            break

# Sonuç mesajı
print(f"Toplam {counter} dosya listelendi ve silindi. Yollar '{output_file}' dosyasina kaydedildi.")
