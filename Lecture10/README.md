# Lecture 10

## İçerik
Bu klasör, Natural Language Processing (NLP - Doğal Dil İşleme) temellerini kapsar.

## Dosyalar
- **Lecture10.pdf**: Dersin teorik sunumu ve ders notları
- **lecture10.py**: NLP preprocessing teknikleri (tokenization, stemming, lemmatization, Bag of Words)
- **lecture10_2.py**: İsim-cinsiyet sınıflandırma uygulaması

## Konu Başlıkları

---

### 1. lecture10.py - NLP Preprocessing ve Feature Extraction

Bu script, metin verilerini makine öğrenmesi için hazırlamak üzere temel NLP tekniklerini gösterir.

#### 1.1. Tokenization (Parçalama)

##### Sentence Tokenization
```python
from nltk.tokenize import sent_tokenize

sample = "This is a sample sentence. This is second sample sentence."
sentences = sent_tokenize(sample)
```

**Çıktı:**
```python
['This is a sample sentence.', 'This is second sample sentence.']
```

**Açıklama:**
- Metni cümlelere böler
- Noktalama işaretlerini (. ! ?) tanır
- "Dr." gibi kısaltmaları da işler (akıllı parsing)
- Kullanım: Doküman analizi, özetleme

##### Word Tokenization
```python
from nltk.tokenize import word_tokenize

words = word_tokenize(sample)
```

**Çıktı:**
```python
['This', 'is', 'a', 'sample', 'sentence', '.', 'This', 'is', 'second', 'sample', 'sentence', '.']
```

**Açıklama:**
- Metni kelimelere ve noktalama işaretlerine böler
- Her noktalama işareti ayrı token
- Boşluklardan daha akıllı (I'm → I, 'm)
- Kullanım: Feature extraction, frequency analysis

#### 1.2. Stemming (Kök Bulma)

Kelimeleri kök formlarına indirger (bazen anlamsız kökler üretir).

##### Porter Stemmer
```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["cared", "university", "fairly", "easily", "singing", "sings", "sung", "singer", "sportingly"]
stemmed_words = [stemmer.stem(word) for word in words]
```

**Çıktı:**
```python
['care', 'univers', 'fairli', 'easili', 'sing', 'sing', 'sung', 'singer', 'sportingli']
```

**Açıklama:**
- **cared → care**: -ed suffix'ini kaldırır
- **university → univers**: -ity suffix'ini kaldırır
- **singing, sings → sing**: -ing, -s kaldırır
- **sportingly → sportingli**: Bazen anlamsız sonuç (hızlı ama hatalı)

**Avantajlar:**
- Hızlı
- Deterministik (aynı input → aynı output)

**Dezavantajlar:**
- Anlamsız kökler üretebilir
- Aynı kökten gelen kelimeler farklı kökler alabilir

##### Lancaster Stemmer
```python
from nltk.stem import LancasterStemmer

lancaster_stemmer = LancasterStemmer()
lancaster_stemmed_words = [lancaster_stemmer.stem(word) for word in words]
```

**Özellikler:**
- Porter'dan daha agresif
- Daha kısa kökler üretir
- Daha fazla hata yapabilir
- Daha hızlı

##### Snowball Stemmer
```python
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
snowball_stemmed_words = [stemmer.stem(word) for word in words]
```

**Özellikler:**
- Porter'ın geliştirilmiş versiyonu
- Çoklu dil desteği (15+ dil)
- Porter'dan daha iyi sonuçlar
- Önerilen: Snowball veya Porter

#### 1.3. Lemmatization (Sözlük Formu Bulma)

Kelimeleri sözlükteki temel formlarına dönüştürür (stemming'den daha doğru).

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
```

**Temel Kullanım:**
```python
['cared', 'university', 'fairly', 'easily', 'singing', 'sings', 'sung', 'singer', 'sportingly']
→
['cared', 'university', 'fairly', 'easily', 'singing', 'sings', 'sung', 'singer', 'sportingly']
```

**Neden değişmedi?**
- Varsayılan POS (Part of Speech) = noun (isim)
- "singing" bir isim olarak değerlendirildi
- Doğru sonuç için POS tag gerekli

##### POS Tag ile Lemmatization
```python
print("better:", lemmatizer.lemmatize("better", pos="a"))   # Adjective (sıfat)
print("rocks:", lemmatizer.lemmatize("rocks", pos="v"))     # Verb (fiil)
print("corpora:", lemmatizer.lemmatize("corpora", pos="n")) # Noun (isim)
```

**Çıktı:**
```python
better: good       # "better" → "good" (sıfat köküne in)
rocks: rock        # "rocks" (fiil) → "rock"
corpora: corpus    # "corpora" (çoğul) → "corpus" (tekil)
```

**POS Etiketleri:**
- **n**: Noun (isim) - varsayılan
- **v**: Verb (fiil)
- **a**: Adjective (sıfat)
- **r**: Adverb (zarf)

**Stemming vs Lemmatization:**

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| **Hız** | Hızlı | Yavaş (sözlük lookup) |
| **Doğruluk** | Düşük | Yüksek |
| **Sonuç** | Kök (anlamsız olabilir) | Sözlük formu (anlamlı) |
| **Örnek** | running → run | running → run (v) / running → running (n) |
| **POS gerekli mi?** | Hayır | Evet (daha iyi sonuç için) |

#### 1.4. Bag of Words (BoW)

Metni sayısal feature vektörüne dönüştürür (kelime frekansları).

```python
from sklearn.feature_extraction.text import CountVectorizer

sentences = [
    "We are using the Bag of Words model.",
    "The Bag of Words Model is used for extracting the features."
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
features_text = X.todense()
print("Features Text:\n", features_text)
print(vectorizer.vocabulary_)
```

**Çıktı:**
```python
Features Text:
[[1 1 0 0 1 1 1 0 1 1 1 0]
 [1 1 1 1 0 1 1 1 0 0 2 1]]

Vocabulary:
{'we': 10, 'are': 0, 'using': 9, 'the': 8, 'bag': 1, 'of': 6, 
 'words': 11, 'model': 5, 'is': 4, 'used': 7, 'for': 3, 
 'extracting': 2, 'features': 4}
```

**Nasıl Çalışır?**

1. **Vocabulary Oluştur:**
   - Tüm benzersiz kelimeleri topla
   - Küçük harfe çevir (lowercase)
   - Alfabetik sırala

2. **Her Cümle için Vector Oluştur:**
   - Vector boyutu = vocabulary boyutu
   - Her pozisyon bir kelimeyi temsil eder
   - Değer = Kelime frekansı

**Örnek:**
```
Vocabulary: ['are', 'bag', 'extracting', 'features', 'for', 'is', 'model', 'of', 'the', 'used', 'using', 'we', 'words']

Sentence 1: "We are using the Bag of Words model."
Vector:     [1,   1,    0,          0,        0,   0,   1,      1,   1,    0,     1,      1,   1]
             are  bag  extracting features for  is  model   of  the  used using   we  words

Sentence 2: "The Bag of Words Model is used for extracting the features."
Vector:     [0,   1,    1,          1,        1,   1,   1,      1,   2,    1,     0,      0,   1]
```

**Özellikler:**
- **Sparse matrix**: Çoğu değer 0 (bellek efficient)
- **N-grams**: Tek kelime yerine 2-3 kelime kombinasyonları
  ```python
  CountVectorizer(ngram_range=(1,2))  # Unigram + bigram
  ```
- **Max features**: Sadece en yaygın N kelimeyi al
  ```python
  CountVectorizer(max_features=100)
  ```

**Dezavantajlar:**
- Kelime sırası kaybolur ("not good" = "good not")
- Yaygın kelimeler (the, is) çok sayılır → TF-IDF kullan
- Büyük vocabulary → Yüksek boyutlu vektör

---

### 2. lecture10_2.py - İsim-Cinsiyet Sınıflandırma

Bu script, isimlerin son harflerine bakarak cinsiyet tahmini yapar.

#### 2.1. Feature Extraction Fonksiyonu

```python
def extract_features(word, N=2):
    last_n_letters = word[-N:]
    return {'feature': last_n_letters.lower()}
```

**Açıklama:**
- `word[-N:]`: Kelimenin son N harfi
  - "David"[-2:] → "id"
  - "Swati"[-2:] → "ti"
- `.lower()`: Küçük harfe çevir
- Dictionary döner: `{'feature': 'id'}`

**Neden son harfler?**
- İngilizce'de isim sonu cinsiyet belirtir:
  - -a, -i: Genelde kadın (Maria, Swati)
  - -n, -d, -r: Genelde erkek (John, David, Peter)
  - -y: Karışık (Mary, Henry)

#### 2.2. Veri Yükleme (NLTK Corpus)

```python
from nltk.corpus import names

male_list = [(name, 'male') for name in names.words('male.txt')]
female_list = [(name, 'female') for name in names.words('female.txt')]
data = (male_list + female_list)
```

**NLTK names corpus:**
- ~8000 isim (male + female)
- male.txt: ~2900 erkek isim
- female.txt: ~5000 kadın isim
- İngilizce isimler

**Veri yapısı:**
```python
[('John', 'male'), ('David', 'male'), ('Mary', 'female'), ...]
```

#### 2.3. Veri Karıştırma

```python
random.seed(5)
random.shuffle(data)
```

**Neden karıştırma?**
- Veri sıralı (önce tüm erkek, sonra tüm kadın)
- Train-test split dengeli olsun
- `seed(5)`: Tekrarlanabilir sonuçlar

#### 2.4. Train-Test Split

```python
train_sample = int(0.8 * len(data))  # %80 eğitim
```

#### 2.5. Model Eğitimi (Loop eksik?)

```python
for i in range(1, 6):
    print("\n Number of letters", i)
    # DEVAM ETMİYOR - Script tamamlanmamış
```

**Eksik olan kodun tahmini tamamlanması:**
```python
for i in range(1, 6):
    print(f"\n Number of letters: {i}")
    
    # Feature extraction
    train = [(extract_features(n, i), gender) for n, gender in data[:train_sample]]
    test = [(extract_features(n, i), gender) for n, gender in data[train_sample:]]
    
    # Model eğitimi
    classifier = NaiveBayesClassifier.train(train)
    
    # Accuracy
    acc = nltk_accuracy(classifier, test)
    print(f"Accuracy: {acc:.2%}")
    
    # Test isimleri
    for name in namesInput:
        features = extract_features(name, i)
        prediction = classifier.classify(features)
        print(f"{name}: {prediction}")
```

**Beklenen Sonuçlar:**
- N=1: Düşük accuracy (~60-70%)
- N=2: Orta accuracy (~75-80%)
- N=3: Yüksek accuracy (~80-85%)
- N=4+: Overfitting riski (çok spesifik)

**Test İsimleri:**
```python
namesInput = ['David', 'Jakob', 'Swati', 'Shubha']
```
- **David**: -id → Erkek (doğru)
- **Jakob**: -ob → Erkek (doğru)
- **Swati**: -ti → Kadın (doğru)
- **Shubha**: -ha → Kadın (doğru)

## Gerekli Kütüphaneler

```python
# Tokenization ve Preprocessing
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer

# Classification
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names

# Diğer
import random
```

**NLTK Data Download:**
```python
import nltk
nltk.download('punkt')        # Tokenizer
nltk.download('wordnet')      # Lemmatizer
nltk.download('omw-1.4')      # Open Multilingual Wordnet
nltk.download('names')        # Names corpus
nltk.download('averaged_perceptron_tagger')  # POS tagger
```

## NLP Pipeline

Tipik bir NLP projesi akışı:

1. **Text Cleaning**
   - Lowercase
   - Noktalama temizleme
   - HTML/URL temizleme

2. **Tokenization**
   - Sentence tokenization
   - Word tokenization

3. **Normalization**
   - Stemming veya Lemmatization
   - Stop words removal (the, is, are)

4. **Feature Extraction**
   - Bag of Words
   - TF-IDF
   - Word Embeddings (Word2Vec, GloVe)

5. **Model Training**
   - Naive Bayes
   - SVM
   - Deep Learning (LSTM, Transformer)

6. **Evaluation**
   - Accuracy, Precision, Recall, F1

## Önemli Kavramlar

### Stop Words
Çok yaygın, az bilgi taşıyan kelimeler:
```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered = [w for w in words if w.lower() not in stop_words]
```

### TF-IDF (Term Frequency - Inverse Document Frequency)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)
```
- Yaygın kelimeler (the, is) düşük ağırlık alır
- Nadir ama önemli kelimeler yüksek ağırlık alır

### N-grams
- **Unigram**: Tek kelime ["I", "love", "NLP"]
- **Bigram**: İki kelime ["I love", "love NLP"]
- **Trigram**: Üç kelime ["I love NLP"]

### POS Tagging
```python
import nltk
text = "Python is amazing"
tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
# [('Python', 'NNP'), ('is', 'VBZ'), ('amazing', 'JJ')]
```

## Naive Bayes Classifier

**Bayes Teoremi:**
```
P(class|features) = P(features|class) × P(class) / P(features)
```

**"Naive" Varsayımı:**
- Tüm feature'lar bağımsızdır
- Gerçekte yanlış, ama pratikte iyi çalışır

**Avantajlar:**
- Hızlı eğitim ve tahmin
- Az veri ile iyi çalışır
- Probabilistic çıktı verir

**Dezavantajlar:**
- Bağımsızlık varsayımı gerçekçi değil
- Zero probability problemi (Laplace smoothing ile çözülür)

## Pratik Uygulamalar

- **Spam Detection**: Email sınıflandırma
- **Sentiment Analysis**: Duygu analizi (pozitif/negatif)
- **Language Detection**: Dil tanıma
- **Named Entity Recognition**: İsim, yer, organizasyon tanıma
- **Machine Translation**: Çeviri
- **Chatbots**: Konuşma botları
- **Text Summarization**: Özetleme

## Notlar

- Stemming hızlı ama hatalı, Lemmatization yavaş ama doğru
- Bag of Words basit ama etkili, kelime sırası kaybolur
- N-grams kelime sırasını kısmen korur ama boyut artar
- Feature extraction, NLP'de en kritik adımdır
- NLTK corpus'ları ilk kullanımda indirilmelidir
