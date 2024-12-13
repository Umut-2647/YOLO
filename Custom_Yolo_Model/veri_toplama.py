import cv2
import os

# Video ve kayıt yolları
video_path = os.path.join(os.path.dirname(__file__), "video.mp4")
save_path = os.path.join(os.path.dirname(__file__), "spot_data/images")

# Kayıt klasörünü oluştur
if not os.path.exists(save_path):
    print("Dosya Bulunamadi")
     
# Video okuma
cap = cv2.VideoCapture(video_path)

i = 8266  # Görüntü dosyalarının başlangıç indexi
frame_index = 0  # Video karelerinin sırası

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Sadece 8. karelerde görüntü kaydet
    if frame_index % 8 == 0:
        # Çözünürlük ayarı
        frame = cv2.resize(frame, (640, 480))
        
        # Görüntüyü kaydet
        cv2.imwrite(os.path.join(save_path, "{}.jpg".format(i)), frame)
        print("{}.jpg kaydedildi.".format(i))
        i += 1

    frame_index += 1

cap.release()
print("Tüm uygun kareler kaydedildi.")
