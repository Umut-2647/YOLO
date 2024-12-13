import cv2
import os
from glob import glob

jpeg=glob(os.path.join(os.path.dirname(__file__), "spot_data/images/*.jpeg"))

for j in jpeg:
    print(j)
    img = cv2.imread(j) # jpeg uzantılı dosyaları oku
    cv2.imwrite(j[:-4]+"jpg", img) # jpeg uzantılı dosyaları jpg'ye çevir
    os.remove(j) # jpeg uzantılı dosyaları sil

print("Tüm jpeg dosyalari jpg'ye çevrildi ve silindi.")  