# python
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Görüntüyü oku (BGR)
img_bgr = cv2.imread('lenna.png')
if img_bgr is None:
    print("Dosya bulunamadı: `lenna.png`")
    sys.exit(1)

# Görüntüyü RGB'ye çevir ve göster
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.show()

# Kaydet (doğru renk için RGB'yi BGR'ye çevirip kaydet)
cv2.imwrite('test_write.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

# GRAYSCALE: doğrudan görüntü dizisini kullan (cv2.imwrite'in dönüşünü değil)
gray_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_image, cmap='gray')
plt.title('Gray Scale Image')
plt.show()


#Convert color space to HSV
hsv_image=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2HSV)
plt.imshow(hsv_image)
plt.title('HSV Image')
plt.show()


#RESIZE: Resize the image to 100x100 pixels
resized_image=cv2.resize(img_rgb,(100,100),interpolation=cv2.INTER_CUBIC)
plt.imshow(resized_image)
size_of_original=img_rgb.shape
plt.title('Resized Image from {} to (200,200)'.format(size_of_original))
plt.show()


#Rotate the image by 90 degrees
(h,w)=img_rgb.shape[:2]
center=(w//2,h//2)
M=cv2.getRotationMatrix2D(center,180,1.0)
rotated_image=cv2.warpAffine(img_rgb,M,(w,h))
plt.imshow(rotated_image)
plt.title('Rotated Image by 180 degrees')
plt.show()

#Image Translation by (100,-100)

M=np.float32([[1,0,100],[0,1,-100]])
shifted_image=cv2.warpAffine(img_rgb,M,(w,h))
plt.imshow(shifted_image)
plt.title('Shifted Image by (100,-100)')
plt.show()

#Edge Detection using Canny
edges=cv2.Canny(img_rgb,100,200)
plt.imshow(edges,cmap='gray')
plt.title('Canny Edge Detection')
plt.show()

#Contour Detection
image_for_contours=cv2.imread('image.jpg')
if image_for_contours is None:
    print("Dosya bulunamadı: `image.jpg`")
    sys.exit(1)
gray_for_contours=cv2.cvtColor(image_for_contours,cv2.COLOR_BGR2GRAY)

ret,thresh=cv2.threshold(gray_for_contours,127,255,0)

contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

contour_image=image_for_contours.copy()
contour_image=cv2.drawContours(contour_image,contours,-1,(0,255,0),5)

plt.imshow(contour_image)
plt.title('Contour Detection')
plt.show()
