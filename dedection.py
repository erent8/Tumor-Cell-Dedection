# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # MRI görüntüsünü yükleme
# image_path = "1.jpg"  # Dosyanızın adını girin
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Gürültüyü azaltmak için Gaussian Blur
# blurred = cv2.GaussianBlur(image, (5,5), 0)

# # Thresholding ile segmentasyon
# _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

# # Kontur bulma
# contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Orijinal görüntüye konturları ve sınır kutusunu çizme
# output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Mavi kutu

# # Görüntüleri görselleştirme
# plt.figure(figsize=(10,5))
# plt.subplot(1,3,1)
# plt.title("Orijinal Görüntü")
# plt.imshow(image, cmap="gray")

# plt.subplot(1,3,2)
# plt.title("Segmentasyon")
# plt.imshow(binary, cmap="gray")

# plt.subplot(1,3,3)
# plt.title("Tespit Edilen Tümör")
# plt.imshow(output)

# plt.show()



import cv2
import numpy as np
import matplotlib.pyplot as plt

# MRI görüntüsünü yükleme
image_path = "1.jpg"  # Dosyanızın adını girin
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Gürültüyü azaltmak için Gaussian Blur
blurred = cv2.GaussianBlur(image, (5,5), 0)

# Thresholding ile segmentasyon
_, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

# Kontur bulma
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Orijinal görüntüye konturları ve sınır kutusunu çizme
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
tumor_areas = []  # Tümörlerin alanlarını saklamak için liste

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    if area > 100:  # Küçük konturları yoksay
        tumor_areas.append(area)
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Yeşil kutular
        cv2.putText(output, f"{area:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

# Görüntüleri görselleştirme
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.title("Orijinal Görüntü")
plt.imshow(image, cmap="gray")

plt.subplot(1,3,2)
plt.title("Segmentasyon")
plt.imshow(binary, cmap="gray")

plt.subplot(1,3,3)
plt.title("Tespit Edilen Tümörler")
plt.imshow(output)

plt.tight_layout()
plt.show()

# Tümörlerin toplam alanını yazdırma
total_area = sum(tumor_areas)
print(f"Tümörlerin toplam alanı: {total_area:.2f} piksel²")
