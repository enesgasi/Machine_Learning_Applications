# Lecture 9

## İçerik
Bu klasör, Audio Processing (Ses İşleme) ve Text-to-Speech (Metinden Sese) teknolojilerini kapsar.

## Dosyalar
- **Lecture9.pdf**: Dersin teorik sunumu ve ders notları
- **lecture9.py**: Ses dosyası okuma, analiz ve üretme
- **lecture9_2.py**: Text-to-Speech (TTS) uygulaması
- **file.wav**: Örnek ses dosyası (input)
- **generated_file.wav**: Üretilen sinüzoidal ses dosyası (output)

## Konu Başlıkları

---

### 1. lecture9.py - Audio Signal Processing

Bu script, ses dosyalarını okuma, analiz etme, yeni ses üretme ve gürültü ekleme işlemlerini gösterir.

#### 1.1. Ses Dosyası Okuma

```python
from scipy.io import wavfile

frequency_sampling, audio_signal = wavfile.read('file.wav')
```

**Dönen Değerler:**
- `frequency_sampling`: Sampling rate (örn: 44100 Hz = CD kalitesi)
  - Saniyede kaç örnek alındığı
  - Yüksek sampling rate = Daha iyi kalite, daha büyük dosya
- `audio_signal`: NumPy array, ses amplitüdleri
  - Mono: 1D array
  - Stereo: 2D array (left, right channels)

#### 1.2. Ses Dosyası Bilgileri

```python
print('Signal shape', audio_signal.shape)
print('Signal data type', audio_signal.dtype)
print('Signal duration', round(audio_signal.shape[0] / float(frequency_sampling), 2))
```

**Çıktı Açıklaması:**
- **Shape**: (n_samples,) mono için, (n_samples, 2) stereo için
- **dtype**: Genelde int16 (-32768 to 32767)
- **Duration**: toplam_sample / sampling_rate = süre (saniye)
  - Örnek: 88200 sample / 44100 Hz = 2 saniye

#### 1.3. Normalizasyon

```python
audio_signal = audio_signal / np.power(2, 15)
```

**Açıklama:**
- int16: -2¹⁵ ile 2¹⁵-1 arası (-32768 to 32767)
- 2¹⁵ = 32768'e böl → [-1, 1] aralığına normalize et
- Float'a çevirerek matematiksel işlemleri kolaylaştırır
- Birçok ses işleme kütüphanesi [-1, 1] bekler

#### 1.4. İlk 100 Sample'ı Görselleştirme

```python
signal = audio_signal[0:100]  # İlk 100 sample
time_axis = 1000 * np.arange(0, len(signal), 1) / float(frequency_sampling)

plt.plot(time_axis, signal, color='red')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Amplitude')
plt.title('Input Audio File')
plt.show()
```

**time_axis Hesabı:**
- `np.arange(0, 100, 1)`: [0, 1, 2, ..., 99]
- `/frequency_sampling`: Saniye cinsine çevir
- `*1000`: Milisaniyeye çevir
- Örnek: 44100 Hz'de 100 sample = 100/44100 = 0.00227 s = 2.27 ms

#### 1.5. Sinüzoidal Ses Üretme

```python
duration = 4                    # 4 saniye
frequency_sampling_2 = 44100    # CD kalitesi
frequency_tone = 784            # Hz (G5 notası, yaklaşık)

min_val = -4 * np.pi
max_val = 4 * np.pi

t = np.linspace(min_val, max_val, duration * frequency_sampling_2)
audio_signal = np.sin(2 * np.pi * frequency_tone * t)
```

**Adım Adım:**

1. **Zaman Vektörü:**
   ```python
   t = np.linspace(-4π, 4π, 4*44100)
   ```
   - 4 saniye × 44100 Hz = 176,400 sample
   - -4π'den 4π'ye eşit aralıklı noktalar

2. **Sinüs Dalgası:**
   ```python
   y = sin(2π × 784 × t)
   ```
   - **2π × frequency**: Radyan/saniye (açısal frekans)
   - **frequency_tone = 784 Hz**: Saniyede 784 tam döngü
   - Amplitude: -1 ile 1 arası

**Neden 2π?**
- sin() fonksiyonu radyan alır
- 1 Hz = 1 cycle/second = 2π radyan/second
- f Hz = 2πf radyan/second

**Müzikal Not:**
- 784 Hz ≈ G5 (sol notası, 5. oktav)
- A4 (la) = 440 Hz (tuning fork standard)
- G5 = A4 × 2^(10/12) ≈ 783.99 Hz

#### 1.6. Ses Dosyası Kaydetme

```python
from scipy.io.wavfile import write

output_file = 'generated_file.wav'
write(output_file, frequency_sampling_2, audio_signal)
```

**Parametreler:**
- **filename**: Kaydedilecek dosya adı
- **rate**: Sampling rate (Hz)
- **data**: Audio array (float veya int)
  - float: [-1, 1] → int16'ya otomatik çevrilir
  - int16: Direkt yazılır

#### 1.7. Gürültü Ekleme

```python
signal = audio_signal[:100]  # İlk 100 sample

rand1 = np.random.uniform(-1.0, 1.0, size=len(signal))
signal = signal + rand1  # Sinyal + White Noise
```

**White Noise:**
- Uniform distribution [-1, 1]
- Her frekans eşit güçte (beyaz ışık gibi)
- SNR (Signal-to-Noise Ratio) düşer
- Görsel: Dalgalanma artar, net sinüs kaybolur

**SNR Hesabı (yapılmamış ama öğretici):**
```python
snr = 10 * np.log10(np.var(audio_signal) / np.var(rand1))
```
- Yüksek SNR: Temiz sinyal
- Düşük SNR: Gürültülü sinyal

---

### 2. lecture9_2.py - Text-to-Speech (TTS)

Bu script, metni sese dönüştürmek için pyttsx3 kütüphanesini kullanır.

#### 2.1. TTS Engine Başlatma

```python
import pyttsx3 as pyt

eng = pyt.init()
```

**pyttsx3:**
- Offline TTS (internet gerektirmez)
- Cross-platform (Windows, Mac, Linux)
- Windows: SAPI5 kullanır
- Mac: NSSpeechSynthesizer
- Linux: espeak

#### 2.2. Engine Ayarları

```python
voices = eng.getProperty('voices')
eng.setProperty('rate', 150)               # Konuşma hızı
eng.setProperty('voice', voices[0].id)     # Ses seçimi
```

**Parametreler:**

- **rate**: Konuşma hızı (words per minute)
  - Varsayılan: ~200 wpm
  - 150: Yavaş ve anlaşılır
  - 100: Çok yavaş
  - 250+: Hızlı, anlaşılması zor

- **voice**: Ses profili
  - `voices[0]`: Genelde erkek ses
  - `voices[1]`: Genelde kadın ses (varsa)
  - Her sesin farklı dil desteği olabilir

**Tüm Sesleri Listeleme:**
```python
for voice in voices:
    print(f"ID: {voice.id}")
    print(f"Name: {voice.name}")
    print(f"Languages: {voice.languages}")
    print(f"Gender: {voice.gender}")
    print("---")
```

#### 2.3. Konuşma Fonksiyonu

```python
def speak(text):
    eng.say(text)        # Metni kuyruğa ekle
    eng.runAndWait()     # Konuşmayı başlat ve bitene kadar bekle
```

**say() vs runAndWait():**
- `say()`: Metni kuyruğa ekler, hemen dönmez
- `runAndWait()`: Kuyruktaki tüm metinleri söyler, blocking
- Birden fazla `say()` çağrısı yapıp sonra `runAndWait()` çağırabilirsin

**Örnek:**
```python
eng.say("First sentence")
eng.say("Second sentence")
eng.say("Third sentence")
eng.runAndWait()  # Üçünü sırayla söyler
```

#### 2.4. Kullanıcı Etkileşimi

```python
speak('what do you want me to say?')

user_input = input('Enter the text you want me to say: ')
speak(user_input)
```

**Akış:**
1. Bilgisayar soruyu sorar (sesli)
2. Kullanıcı terminal'e metin girer
3. Bilgisayar girilen metni okur (sesli)

## Gerekli Kütüphaneler

### lecture9.py
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.io import wavfile
import random
```

### lecture9_2.py
```python
import pyttsx3 as pyt
import engineio as eng  # Not: Bu gereksiz görünüyor, muhtemelen hata
```

**Kurulum:**
```bash
pip install scipy matplotlib pandas
pip install pyttsx3

# Linux için ek:
sudo apt-get install espeak
```

## Ses İşleme Terimleri

### Sampling Rate (Örnekleme Frekansı)
- Saniyede alınan örnek sayısı
- **44100 Hz**: CD kalitesi
- **48000 Hz**: DVD/Video standard
- **22050 Hz**: Radyo kalitesi
- **8000 Hz**: Telefon kalitesi

**Nyquist Teoremi:**
- Sampling rate ≥ 2 × max_frequency
- 44100 Hz → 22050 Hz'e kadar frekansları yakalayabilir
- İnsan kulağı: ~20 Hz - 20000 Hz

### Amplitude (Genlik)
- Ses dalgasının yüksekliği
- Yüksek amplitude = Yüksek ses
- Normalized: [-1, 1] veya [0, 1]
- Int16: [-32768, 32767]

### Frequency (Frekans)
- Saniyede kaç döngü (Hz)
- Düşük Hz = Bas sesler
- Yüksek Hz = Tiz sesler
- 440 Hz = A4 notası (standart tuning)

### Waveform (Dalga Formu)
- **Sine Wave**: Saf ton, tek frekans
- **Square Wave**: Keskin, robot sesi
- **Sawtooth Wave**: Testere dişi, kaba ses
- **White Noise**: Tüm frekanslar eşit

## Ses Formatları

| Format | Özellik | Kullanım |
|--------|---------|----------|
| **WAV** | Sıkıştırılmamış, yüksek kalite | Profesyonel, editing |
| **MP3** | Lossy sıkıştırma, küçük boyut | Müzik paylaşımı |
| **FLAC** | Lossless sıkıştırma | Arşivleme |
| **AAC** | Lossy, MP3'ten iyi | Streaming |

## Pratik Uygulamalar

### Ses Analizi
- Konuşma tanıma (Speech Recognition)
- Müzik türü sınıflandırma
- Duygusal analiz (öfke, mutluluk)
- Pitch detection (oto-tune)

### Ses Sentezleme
- Text-to-Speech (TTS)
- Müzik üretimi (synthesizer)
- Ses efektleri (reverb, echo)
- Voice changer

### Gürültü Azaltma
- Low-pass filter (yüksek frekansları kes)
- High-pass filter (düşük frekansları kes)
- Band-pass filter (belirli aralığı tut)
- Spectral subtraction

## Görselleştirme Tipleri

1. **Waveform**: Zaman vs Amplitude
   - En basit görselleştirme
   - Genel pattern'i gösterir

2. **Spectrogram**: Zaman vs Frekans vs Güç
   - Hangi frekanslarin ne zaman aktif olduğunu gösterir
   - Renkli heat map
   - Konuşma analizinde kritik

3. **Spectrum**: Frekans vs Güç
   - Belirli bir an için frekans dağılımı
   - FFT (Fast Fourier Transform) ile hesaplanır

## İleri Seviye Konular

### Fourier Transform
```python
from scipy.fft import fft, fftfreq

yf = fft(audio_signal)
xf = fftfreq(len(audio_signal), 1/frequency_sampling)

plt.plot(xf, np.abs(yf))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
```

### Mel Spectrogram (ML için)
```python
import librosa

mel_spec = librosa.feature.melspectrogram(y=audio_signal, sr=frequency_sampling)
log_mel = librosa.power_to_db(mel_spec, ref=np.max)
```
- İnsan kulağının frekans algısına uygun
- Speech recognition ve music classification için kullanılır

## Notlar

- WAV dosyaları sıkıştırılmamıştır, dosya boyutu büyüktür
- pyttsx3 offline çalışır, internet bağlantısı gerektirmez
- Gürültü ekleme, ses işleme algoritmalarını test için kullanılır
- Sinüzoidal ses, belirli frekansları test etmek için idealdir
- Text-to-Speech, erişilebilirlik uygulamalarında kritiktir
