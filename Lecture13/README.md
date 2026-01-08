# Lecture 13

## İçerik
Bu klasör, Computer Vision (Bilgisayarlı Görü) ve Image Processing (Görüntü İşleme) temellerini kapsar.

## Dosyalar
- **Lecture13.pdf**: Dersin teorik sunumu ve ders notları
- **lecture13.py**: OpenCV ile görüntü işleme operasyonları
- **Lenna.png**: Test görüntüsü (Lena Söderberg - klasik CV test image)
- **image.jpg**: Contour detection için kullanılan görüntü

## Konu Başlıkları

---

### Computer Vision ve OpenCV

**OpenCV (Open Source Computer Vision Library):**
- C++, Python, Java desteği
- 2500+ optimize edilmiş algoritma
- Gerçek zamanlı bilgisayarlı görü
- BSD lisansı (ticari kullanım ücretsiz)

**Kullanım Alanları:**
- Yüz tanıma
- Nesne tespiti
- Hareket takibi
- Optik karakter tanıma (OCR)
- Otonom araçlar
- Medikal görüntü analizi

---

### lecture13.py - Görüntü İşleme Operasyonları

#### 1. Görüntü Okuma ve RGB Dönüşümü

```python
import cv2
import sys

img_bgr = cv2.imread('lenna.png')
if img_bgr is None:
    print("Dosya bulunamadı: `lenna.png`")
    sys.exit(1)
```

**cv2.imread():**
- Görüntüyü BGR formatında okur (OpenCV default)
- None döner if dosya yoksa
- Flags:
  - `cv2.IMREAD_COLOR`: Renkli (default)
  - `cv2.IMREAD_GRAYSCALE`: Gri tonlamalı
  - `cv2.IMREAD_UNCHANGED`: Alpha channel dahil

**BGR vs RGB:**
- **OpenCV**: BGR (Blue, Green, Red)
- **Matplotlib/PIL**: RGB (Red, Green, Blue)
- **Neden BGR?** Tarihsel nedenler (eski kameralar)

#### RGB Dönüşümü
```python
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

import matplotlib.pyplot as plt
plt.imshow(img_rgb)
plt.title('Original Image')
plt.show()
```

**cv2.cvtColor():**
- Renk uzayı dönüşümleri
- BGR → RGB, RGB → HSV, RGB → Grayscale, vb.
- `COLOR_BGR2RGB`: BGR'yi RGB'ye çevir

**Matplotlib ile Görselleştirme:**
- `plt.imshow()`: Görüntüyü göster
- RGB format bekler (BGR gösterirse renkler yanlış)
- Örnek: Kırmızı → Mavi görünür

#### 2. Görüntü Kaydetme

```python
cv2.imwrite('test_write.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
```

**Açıklama:**
1. `cv2.COLOR_RGB2BGR`: RGB'yi tekrar BGR'ye çevir
2. `cv2.imwrite()`: Dosyaya yaz
   - BGR format bekler
   - Format, uzantıdan anlaşılır (.jpg, .png, .bmp)

**Dosya Formatları:**
- **JPEG**: Lossy sıkıştırma, fotoğraflar için
- **PNG**: Lossless, şeffaflık desteği
- **BMP**: Sıkıştırmasız, büyük boyut
- **TIFF**: Yüksek kalite, bilimsel

#### 3. Grayscale (Gri Tonlama) Dönüşümü

```python
gray_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_image, cmap='gray')
plt.title('Gray Scale Image')
plt.show()
```

**Grayscale Formülü:**
```
Gray = 0.299×R + 0.587×G + 0.114×B
```

**Açıklama:**
- İnsan gözü yeşili daha iyi algılar (0.587 ağırlık)
- RGB → Tek kanal (3D → 2D)
- Shape: (height, width, 3) → (height, width)

**Neden Grayscale?**
- Daha hızlı işleme (3× az veri)
- Birçok algoritma renk gerektirmez (edge detection)
- Bellek tasarrufu

**cmap='gray':**
- Matplotlib colormap'i
- Olmadan: Yeşil-sarı tonlarda gösterir
- 'gray' ile: Siyah-beyaz tonlarda gösterir

#### 4. HSV Renk Uzayı

```python
hsv_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
plt.imshow(hsv_image)
plt.title('HSV Image')
plt.show()
```

**HSV (Hue, Saturation, Value):**
- **Hue (Ton)**: Renk türü (0-179°)
  - 0° = Kırmızı
  - 60° = Sarı
  - 120° = Yeşil
  - 180° = Cyan (mavi-yeşil)
- **Saturation (Doygunluk)**: Renk canlılığı (0-255)
  - 0 = Gri
  - 255 = Çok canlı
- **Value (Parlaklık)**: Işık miktarı (0-255)
  - 0 = Siyah
  - 255 = Parlak

**Neden HSV?**
- Renk segmentasyonu daha kolay
- Işık değişimlerine robust
- İnsan algısına yakın

**Kullanım Örneği:**
```python
# Mavi rengi algıla
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
```

#### 5. Resize (Yeniden Boyutlandırma)

```python
resized_image = cv2.resize(img_rgb, (100, 100), interpolation=cv2.INTER_CUBIC)
plt.imshow(resized_image)
size_of_original = img_rgb.shape
plt.title(f'Resized Image from {size_of_original} to (100,100)')
plt.show()
```

**Parametreler:**
- **(100, 100)**: Yeni boyut (width, height) - **DİKKAT: Ters sıra!**
- **interpolation**: Piksel değerlerini nasıl hesapla

**Interpolation Metodları:**

| Method | Hız | Kalite | Kullanım |
|--------|-----|--------|----------|
| `INTER_NEAREST` | En hızlı | En düşük | Küçültme, hız kritik |
| `INTER_LINEAR` | Hızlı | Orta | Genel amaçlı |
| `INTER_CUBIC` | Yavaş | Yüksek | Büyütme, kalite kritik |
| `INTER_LANCZOS4` | En yavaş | En yüksek | Profesyonel |

**INTER_CUBIC:**
- Bicubic interpolation
- 4×4 piksel komşuluğu kullanır
- Smooth geçişler
- Büyültmede önerilen

**Aspect Ratio:**
```python
# Oranı koru
h, w = img_rgb.shape[:2]
new_width = 200
new_height = int(h * (new_width / w))
resized = cv2.resize(img_rgb, (new_width, new_height))
```

#### 6. Rotation (Döndürme)

```python
(h, w) = img_rgb.shape[:2]  # Height, Width
center = (w // 2, h // 2)   # Merkez nokta

M = cv2.getRotationMatrix2D(center, 180, 1.0)
rotated_image = cv2.warpAffine(img_rgb, M, (w, h))

plt.imshow(rotated_image)
plt.title('Rotated Image by 180 degrees')
plt.show()
```

**cv2.getRotationMatrix2D():**
```python
M = cv2.getRotationMatrix2D(center, angle, scale)
```
- **center**: Dönme merkezi (x, y)
- **angle**: Derece (pozitif = saat yönü tersı)
- **scale**: Ölçek faktörü (1.0 = aynı boyut)

**Dönüşüm Matrisi M:**
```
M = [cos(θ)  -sin(θ)   tx]
    [sin(θ)   cos(θ)   ty]
```
- 2×3 affine transformation matrisi
- tx, ty: Translation (kayma)

**cv2.warpAffine():**
```python
result = cv2.warpAffine(src, M, (width, height))
```
- Affine transformation uygula
- Görüntüyü M matrisine göre dönüştür
- Output size: (width, height)

**Örnek Açılar:**
- 90°: Saat yönü tersine çeyrek dönüş
- 180°: Ters çevir (baş aşağı)
- 270° or -90°: Saat yönünde çeyrek dönüş

#### 7. Translation (Kaydırma)

```python
M = np.float32([[1, 0, 100], [0, 1, -100]])
shifted_image = cv2.warpAffine(img_rgb, M, (w, h))

plt.imshow(shifted_image)
plt.title('Shifted Image by (100,-100)')
plt.show()
```

**Translation Matrix:**
```
M = [1  0  tx]
    [0  1  ty]
```
- **tx**: Yatay kayma (pozitif = sağa, negatif = sola)
- **ty**: Dikey kayma (pozitif = aşağı, negatif = yukarı)

**Örnek:**
```python
M = [[1, 0, 100], [0, 1, -100]]
```
- tx = 100: 100 piksel sağa
- ty = -100: 100 piksel yukarı

**np.float32():**
- OpenCV float32 matrix bekler
- Python list'i numpy array'e çevir

#### 8. Edge Detection (Kenar Tespiti)

```python
edges = cv2.Canny(img_rgb, 100, 200)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.show()
```

**Canny Edge Detector:**
- 1986'da John Canny tarafından geliştirildi
- Multi-stage algoritma
- En popüler edge detection yöntemi

**cv2.Canny() Parametreleri:**
```python
edges = cv2.Canny(image, threshold1, threshold2, apertureSize=3, L2gradient=False)
```

**Thresholds:**
- **threshold1 (100)**: Lower threshold
  - Düşük gradient → Kenar değil
- **threshold2 (200)**: Upper threshold
  - Yüksek gradient → Kesin kenar
- **Arası**: Bağlantılıysa kenar

**Algoritma Adımları:**

1. **Gaussian Blur**: Gürültü azalt
2. **Gradient Hesapla**: Sobel operatörü ile
3. **Non-maximum Suppression**: İnce kenarlar
4. **Double Threshold**: İki eşik uygula
5. **Edge Tracking**: Hysteresis ile kenarları izle

**Örnek:**
```
Gradient magnitude:
50   90  150  180  120   80
    ↓    ↓    ↓    ↓    ↓
threshold1=100, threshold2=200:
0    0   Edge Edge  0    0
      (Weak)(Strong)
```

**Kullanım İpuçları:**
- threshold2 / threshold1 ≈ 2:1 veya 3:1 oranı önerilen
- Düşük threshold: Fazla kenar (gürültü)
- Yüksek threshold: Az kenar (detay kaybı)

#### 9. Contour Detection (Kontur Tespiti)

```python
image_for_contours = cv2.imread('image.jpg')
if image_for_contours is None:
    print("Dosya bulunamadı: `image.jpg`")
    sys.exit(1)

gray_for_contours = cv2.cvtColor(image_for_contours, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_for_contours, 127, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contour_image = image_for_contours.copy()
contour_image = cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 5)

plt.imshow(contour_image)
plt.title('Contour Detection')
plt.show()
```

**Adım Adım:**

##### 9.1. Thresholding (Eşikleme)
```python
ret, thresh = cv2.threshold(gray_for_contours, 127, 255, 0)
```

**Parametreler:**
- **gray_for_contours**: Gri tonlamalı görüntü
- **127**: Threshold değeri
- **255**: Maksimum değer
- **0**: Thresholding türü (THRESH_BINARY)

**Binary Thresholding:**
```python
if pixel > 127:
    pixel = 255  # Beyaz
else:
    pixel = 0    # Siyah
```

**Sonuç:** Binary (siyah-beyaz) görüntü

**Diğer Threshold Türleri:**
- `THRESH_BINARY_INV`: Ters binary
- `THRESH_TRUNC`: Truncate (kırp)
- `THRESH_TOZERO`: Sıfıra indir
- `THRESH_OTSU`: Otomatik threshold (Otsu's method)

##### 9.2. Contour Bulma
```python
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

**Dönen Değerler:**
- **contours**: List of contours
  - Her contour: NumPy array of (x, y) coordinates
- **hierarchy**: Contour hiyerarşisi
  - [Next, Previous, First_Child, Parent]

**cv2.RETR_TREE:**
- Retrieval mode
- Tüm contour'ları hiyerarşik ağaç yapısında bul
- Parent-child ilişkilerini koru

**Retrieval Modes:**
- `RETR_EXTERNAL`: Sadece dış contour'lar
- `RETR_LIST`: Tüm contour'lar, hiyerarşisiz
- `RETR_TREE`: Tam hiyerarşi (bu kodda)
- `RETR_CCOMP`: 2 seviye hiyerarşi

**cv2.CHAIN_APPROX_SIMPLE:**
- Contour approximation method
- Gereksiz noktaları kaldır
- Örnek: Düz çizgi → Sadece uç noktalar

**Approximation Methods:**
- `CHAIN_APPROX_NONE`: Tüm noktalar
- `CHAIN_APPROX_SIMPLE`: Basitleştirilmiş (bu kodda)
- `CHAIN_APPROX_TC89_L1`: Teh-Chin chain approx

**Örnek:**
```python
# 4 köşeli dikdörtgen
CHAIN_APPROX_NONE: 100+ nokta (her piksel)
CHAIN_APPROX_SIMPLE: 4 nokta (sadece köşeler)
```

##### 9.3. Contour Çizme
```python
contour_image = image_for_contours.copy()  # Orjinali bozma
contour_image = cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 5)
```

**cv2.drawContours():**
```python
cv2.drawContours(image, contours, contourIdx, color, thickness)
```

**Parametreler:**
- **image**: Üzerine çizilecek görüntü
- **contours**: Contour listesi
- **contourIdx**: Hangi contour çizilsin?
  - -1: Hepsini çiz
  - 0, 1, 2...: Belirli index
- **color**: (B, G, R) renk tuple'ı
  - (0, 255, 0): Yeşil
- **thickness**: Çizgi kalınlığı
  - -1 veya cv2.FILLED: İçini doldur

**Contour İşlemleri:**
```python
# Contour alanı
area = cv2.contourArea(contours[0])

# Contour uzunluğu (perimeter)
perimeter = cv2.arcLength(contours[0], True)

# Bounding rectangle
x, y, w, h = cv2.boundingRect(contours[0])
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Minimum enclosing circle
(x, y), radius = cv2.minEnclosingCircle(contours[0])
cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 0), 2)
```

## Gerekli Kütüphaneler

```python
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
```

**Kurulum:**
```bash
pip install opencv-python
pip install matplotlib
pip install numpy
```

## OpenCV Temel Veri Yapısı

**Görüntü = NumPy Array**

```python
img.shape  # (height, width, channels)
```

**Grayscale:**
```python
img.shape = (512, 512)      # 2D array
img.dtype = uint8           # 0-255
```

**RGB/BGR:**
```python
img.shape = (512, 512, 3)   # 3D array
img[:, :, 0]                # Blue channel
img[:, :, 1]                # Green channel
img[:, :, 2]                # Red channel
```

## Koordinat Sistemi

**OpenCV:**
```
(0,0) ──────► x (width)
  │
  │
  ▼
  y (height)
```

**Piksel Erişimi:**
```python
pixel = img[y, x]           # NOT: [row, col] = [y, x]
pixel = img[100, 200]       # 100. satır, 200. sütun
```

**Dikkat:** (x, y) sırası fonksiyona göre değişir!
- `cv2.circle(img, (x, y), ...)`: (x, y)
- `img[y, x]`: [y, x]

## Image Processing Pipeline

Tipik bir bilgisayarlı görü projesi:

1. **Image Acquisition**: Görüntü al (camera/file)
2. **Preprocessing**: Gri tonlama, resize, blur
3. **Feature Extraction**: Edge, corner, contour
4. **Segmentation**: Nesne/arkaplan ayırma
5. **Object Detection**: Nesne konum tespiti
6. **Classification**: Nesne tanıma
7. **Post-processing**: Sonuçları görselleştir

## Renk Uzayları Karşılaştırması

| Uzay | Kullanım | Avantaj | Dezavantaj |
|------|----------|---------|------------|
| **RGB** | Genel amaçlı | Yaygın | Işığa duyarlı |
| **Grayscale** | Basit işlemler | Hızlı | Renk yok |
| **HSV** | Renk segmentasyonu | Robust | Dönüşüm gerekir |
| **LAB** | Perceptual | İnsan gözüne yakın | Karmaşık |
| **YCrCb** | Video compression | Efficient | Spesifik |

## Görüntü Filtreleme

### Blurring (Bulanıklaştırma)
```python
# Gaussian Blur
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Median Blur
blurred = cv2.medianBlur(img, 5)

# Bilateral Filter (kenarları korur)
blurred = cv2.bilateralFilter(img, 9, 75, 75)
```

### Sharpening (Keskinleştirme)
```python
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
sharpened = cv2.filter2D(img, -1, kernel)
```

## Pratik Uygulamalar

- **Face Detection**: Haar Cascades, MTCNN
- **Object Detection**: YOLO, SSD, Faster R-CNN
- **Image Segmentation**: U-Net, Mask R-CNN
- **OCR**: Tesseract, EasyOCR
- **Image Classification**: ResNet, VGG, EfficientNet
- **Style Transfer**: Neural style transfer
- **Image Enhancement**: Super-resolution

## Notlar

- OpenCV BGR kullanır, Matplotlib RGB bekler (dikkat!)
- `img.shape`: (height, width, channels) - Width ve Height ters!
- Affine transformation: Rotation, translation, scaling
- Perspective transformation: 3D → 2D projection
- Contour'lar şekil analizi için güçlüdür
- Thresholding, edge detection öncesi yapılmalı
- cv2.imshow() yerine plt.imshow() kullanılmış (Jupyter uyumlu)
- Lenna görüntüsü, image processing'de standart test görüntüsüdür
