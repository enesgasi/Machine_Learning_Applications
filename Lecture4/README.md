# Lecture 4

## İçerik
Bu klasör, sınıflandırma algoritmalarına odaklanan ders materyallerini içerir.

## Dosyalar
- **Lecture4.pdf**: Dersin teorik sunumu ve ders notları
- **lecture4.py**: Logistic Regression ve Naive Bayes uygulamaları

## Konu Başlıkları

### 1. Logistic Regression (Lojistik Regresyon)
- İkili ve Çoklu Sınıflandırma
- Karar Sınırları (Decision Boundaries)
- Sigmoid Fonksiyonu
- Model Parametreleri ve Solver Seçimleri
- Görselleştirme Teknikleri

### 2. Naive Bayes Classifier
- Gaussian Naive Bayes
- Olasılıksal Sınıflandırma
- Breast Cancer Dataset Uygulaması

### 3. Model Değerlendirme Metrikleri
- **Accuracy (Doğruluk)**: Doğru tahminlerin oranı
- **Precision (Kesinlik)**: Pozitif tahminlerin doğruluğu
- **Recall (Duyarlılık)**: Gerçek pozitifleri bulma oranı
- **F1-Score**: Precision ve Recall'ın harmonik ortalaması
- **Confusion Matrix**: Tahmin performansını detaylı inceleme

## Kod Örnekleri

### Logistic Regression Örneği
```python
from sklearn import linear_model
Classifier_LR = linear_model.LogisticRegression(solver='liblinear', C=75)
Classifier_LR.fit(X, y)
```

### Naive Bayes Örneği
```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(train, train_labels)
preds = gnb.predict(test)
```

## Veri Setleri
- **Breast Cancer Dataset**: Meme kanseri teşhisi için kullanılan sklearn'in yerleşik veri seti

## Gerekli Kütüphaneler
```python
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
```

## Çıktılar ve Sonuçlar
Script, farklı solver'lar ve parametrelerle Logistic Regression modellerinin performansını karşılaştırır ve karar sınırlarını görselleştirir.
