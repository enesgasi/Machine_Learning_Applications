# Lecture 5

## İçerik
Bu klasör, Decision Tree (Karar Ağaçları) algoritması üzerine ders materyallerini içerir.

## Dosyalar
- **Lecture5.pdf**: Dersin teorik sunumu ve ders notları
- **Lecture5_2.py**: Decision Tree sınıflandırma uygulaması

## Konu Başlıkları

### 1. Decision Tree (Karar Ağacı) Algoritması
- Karar ağacı mantığı ve çalışma prensibi
- Node, Branch ve Leaf kavramları
- Feature importance
- Gini Index ve Entropy
- Pruning (Budama) teknikleri

### 2. Cinsiyet Tahmin Uygulaması
Script, kişilerin boy uzunluğu ve saç uzunluğu verilerini kullanarak cinsiyetlerini tahmin eden bir model oluşturur.

### 3. Model Görselleştirme
- Graphviz ve Pydotplus kullanarak karar ağacının görsel temsili
- PNG formatında ağaç diyagramı oluşturma
- Node'ların renklendirilerek kategorilere göre gösterimi

## Kod Yapısı

### Veri Hazırlama
```python
X=[[165,19],[175,32],[136,35],[174,65],...]  # [boy, saç uzunluğu]
Y =['Man','Woman','Woman','Man',...]  # Etiketler
```

### Model Eğitimi
```python
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
prediction = clf.predict([[163,15]])  # Tahmin
```

### Görselleştirme
```python
import pydotplus
dot_data = tree.export_graphviz(clf, feature_names=['height','length of hair'], 
                                 out_file=None, filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('Decisiontree2.png')
```

## Gerekli Kütüphaneler
```bash
conda install -c conda-forge pydotplus
conda install -c anaconda graphviz
```

```python
import pydotplus
from sklearn import tree
from sklearn.model_selection import train_test_split
import collections
```

## Veri Özellikleri
- **height**: Boy uzunluğu (cm)
- **length of hair**: Saç uzunluğu (cm)
- **Target**: Cinsiyet (Man/Woman)

## Çıktılar
- **Decisiontree2.png**: Karar ağacının görsel temsili
- Terminal'de tahmin sonuçları

## Train-Test Split
Model, %40 test verisi ile değerlendirilmek üzere train-test split yapılarak eğitilir.
