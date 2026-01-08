# Lecture 11

## İçerik
Bu klasör, Genetic Algorithms (Genetik Algoritmalar) konusunu kapsar.

## Dosyalar
- **Lecture11.pdf**: Dersin teorik sunumu ve ders notları
- **lecture11.py**: Genetic Algorithm ile optimizasyon problemi çözümü

## Konu Başlıkları

---

### Genetic Algorithm Nedir?

Genetic Algorithm (GA), doğal seçilim ve evrim prensiplerinden esinlenen bir optimizasyon algoritmasıdır.

**Temel Kavramlar:**
- **Chromosome (Kromozom)**: Bir çözüm (weight vektörü)
- **Gene (Gen)**: Bir parametr value (tek bir weight)
- **Population (Popülasyon)**: Çözümler topluluğu
- **Fitness**: Çözümün kalitesi (ne kadar iyi?)
- **Generation (Nesil)**: Evrim iterasyonu

**Evrim Süreci:**
1. **Initialization**: Rastgele popülasyon oluştur
2. **Selection**: En iyi bireyleri seç (elitizm)
3. **Crossover**: Seçilenleri çiftleştir, offspring üret
4. **Mutation**: Rastgele değişiklikler yap
5. **Replacement**: Yeni nesil oluştur
6. **Repeat**: Yeterli fitness'a ulaşana kadar tekrarla

---

### lecture11.py - Lineer Denklem Optimizasyonu

#### Problem Tanımı

**Amaç:**
```python
y = w₁×4 + w₂×(-2) + w₃×3.5 + w₄×5 + w₅×(-11) + w₆×(-4.7)
```
Bu denklemi maksimize eden w₁, w₂, w₃, w₄, w₅, w₆ değerlerini bul.

**Örnek:**
```python
equation_inputs = [4, -2, 3.5, 5, -11, -4.7]
```
- w₁ = 4 olsun → 4×4 = 16 (pozitif katkı)
- w₂ = -2 olsun → -2×(-2) = 4 (pozitif katkı)
- w₅ = -11 olsun → -11×(-11) = 121 (pozitif katkı)

**Sezgisel Çözüm:**
- Pozitif coefficient'leri büyük pozitif w'lerle çarp
- Negatif coefficient'leri büyük negatif w'lerle çarp (eksi × eksi = artı)
- Örnek: w = [4, 4, 4, 4, 4, 4] → Fitness düşük olur (negatifler çarpmak kötü)

#### 1. Parametreler ve Başlangıç

```python
equation_inputs = [4, -2, 3.5, 5, -11, -4.7]
num_weights = 6                 # Kaç gene (w₁...w₆)
sol_per_population = 8           # Popülasyon boyutu (8 birey)
num_parents_mating = 4           # Kaç parent seçilecek (en iyi 4)
pop_size = (sol_per_population, num_weights)  # (8, 6) matris
```

**İlk Popülasyon:**
```python
new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_size)
```
- 8 birey, her biri 6 gen
- Her gen [-4, 4] arasında rastgele değer
- Shape: (8, 6)

**Örnek Popülasyon:**
```
Individual 0: [2.3, -1.5, 0.8, 3.2, -0.9, 1.7]
Individual 1: [-2.1, 3.4, -1.2, 0.6, 2.8, -3.5]
...
Individual 7: [1.1, -0.3, 2.9, -1.8, 0.5, 3.3]
```

#### 2. Fitness Fonksiyonu

```python
def calc_pop_fitness(equation_inputs, pop):
    fitness = np.sum(np.array(equation_inputs) * pop, axis=1)
    return fitness
```

**Nasıl Çalışır:**

1. **Broadcasting:**
   ```python
   equation_inputs = [4, -2, 3.5, 5, -11, -4.7]
   pop = [[2.3, -1.5, 0.8, 3.2, -0.9, 1.7],
          [1.2, 0.5, -2.1, 1.8, 3.4, -1.9],
          ...]
   ```

2. **Element-wise Multiplication:**
   ```python
   [4, -2, 3.5, 5, -11, -4.7] × [2.3, -1.5, 0.8, 3.2, -0.9, 1.7]
   = [9.2, 3.0, 2.8, 16.0, 9.9, -7.99]
   ```

3. **Sum (axis=1):**
   ```python
   fitness = 9.2 + 3.0 + 2.8 + 16.0 + 9.9 + (-7.99) = 32.91
   ```

4. **Tüm Popülasyon:**
   ```python
   fitness = [32.91, 18.5, 45.2, 12.8, 38.6, 29.1, 41.3, 22.7]
   ```

**Yüksek fitness = İyi çözüm**

#### 3. Parent Selection (Elitism)

```python
def select_mating_pool(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    fitness_copy = fitness.copy()
    
    for parent_num in range(num_parents):
        max_idx = np.argmax(fitness_copy)          # En iyi fitness'ın index'i
        parents[parent_num, :] = pop[max_idx, :]   # O bireyi parent yap
        fitness_copy[max_idx] = -np.inf            # Tekrar seçilmesin
    return parents
```

**Adım Adım:**

1. **İlk Parent:**
   - fitness = [32.91, 18.5, **45.2**, 12.8, 38.6, 29.1, 41.3, 22.7]
   - max_idx = 2 (index 2'de 45.2 var)
   - parents[0] = pop[2] (en iyi birey)
   - fitness_copy[2] = -∞ (işaretle)

2. **İkinci Parent:**
   - fitness_copy = [32.91, 18.5, -∞, 12.8, 38.6, 29.1, **41.3**, 22.7]
   - max_idx = 6 (index 6'da 41.3 var)
   - parents[1] = pop[6] (2. en iyi)
   - fitness_copy[6] = -∞

3. **Üçüncü Parent:**
   - fitness_copy = [32.91, 18.5, -∞, 12.8, **38.6**, 29.1, -∞, 22.7]
   - max_idx = 4
   - parents[2] = pop[4]

4. **Dördüncü Parent:**
   - fitness_copy = [**32.91**, 18.5, -∞, 12.8, -∞, 29.1, -∞, 22.7]
   - max_idx = 0
   - parents[3] = pop[0]

**Sonuç:**
```python
parents = [pop[2], pop[6], pop[4], pop[0]]  # En iyi 4 birey
```

#### 4. Crossover (Çaprazlama)

```python
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = offspring_size[1] // 2  # Ortadan böl
    
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        
        offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    
    return offspring
```

**Parametreler:**
```python
offspring_size = (sol_per_population - parents.shape[0], num_weights)
                = (8 - 4, 6) = (4, 6)
```
- 4 offspring üretilecek (8 total - 4 parent = 4 yeni)

**Crossover Point:**
```python
crossover_point = 6 // 2 = 3
```
- İlk 3 gen birinci parent'tan
- Son 3 gen ikinci parent'tan

**Örnek:**

```
Parent 0: [2.3, -1.5,  0.8 | 3.2, -0.9, 1.7]
Parent 1: [1.1, -0.3,  2.9 | -1.8, 0.5, 3.3]
          └─────────┘   └─────────┘
          İlk 3 gen     Son 3 gen

Offspring 0: Parent0[:3] + Parent1[3:] = [2.3, -1.5, 0.8 | -1.8, 0.5, 3.3]
```

**Parent Kombinasyonları:**
```python
k=0: parent1_idx = 0 % 4 = 0, parent2_idx = 1 % 4 = 1 → Parent0 + Parent1
k=1: parent1_idx = 1 % 4 = 1, parent2_idx = 2 % 4 = 2 → Parent1 + Parent2
k=2: parent1_idx = 2 % 4 = 2, parent2_idx = 3 % 4 = 3 → Parent2 + Parent3
k=3: parent1_idx = 3 % 4 = 3, parent2_idx = 4 % 4 = 0 → Parent3 + Parent0
```

**Crossover Türleri (Bu kodda single-point):**
- **Single-point**: Tek noktadan böl
- **Two-point**: İki noktadan böl
- **Uniform**: Her gen için rastgele seç
- **Arithmetic**: Ortalamasını al

#### 5. Mutation (Mutasyon)

```python
def mutation(offspring):
    for idx in range(offspring.shape[0]):
        random_value = float(np.random.uniform(-1.0, 1.0))
        offspring[idx, 4] = offspring[idx, 4] + random_value
    return offspring
```

**Açıklama:**
- Her offspring için 5. gen (index 4)'i mutate et
- Rastgele [-1, 1] değer ekle
- Diğer genler değişmez

**Örnek:**
```python
Before: [2.3, -1.5, 0.8, -1.8, 0.5, 3.3]
Random: 0.73
After:  [2.3, -1.5, 0.8, -1.8, 1.23, 3.3]
                              ↑
                          Mutate edildi
```

**Neden Mutasyon?**
- **Diversity**: Popülasyona çeşitlilik katar
- **Local Optimum'dan Kaçış**: Sıkışmayı engeller
- **Exploration**: Yeni çözüm alanları keşfeder

**Mutasyon Türleri:**
- **Random mutation**: Bu kodda kullanılan
- **Gaussian mutation**: Normal dağılımdan ekle
- **Swap mutation**: İki gen'i takas et

#### 6. Yeni Nesil Oluşturma (Elitism)

```python
new_population[0:parents.shape[0], :] = parents              # İlk 4: Parent'lar
new_population[parents.shape[0]:, :] = offspring_mutation    # Son 4: Offspring'ler
```

**Elitism:**
- En iyi bireyleri direk bir sonraki nesle aktar
- Garantili monoton iyileşme
- Best solution hiçbir zaman kaybolmaz

**Yeni Popülasyon:**
```
[0] Parent 0 (en iyi)
[1] Parent 1 (2. en iyi)
[2] Parent 2 (3. en iyi)
[3] Parent 3 (4. en iyi)
[4] Offspring 0 (mutated)
[5] Offspring 1 (mutated)
[6] Offspring 2 (mutated)
[7] Offspring 3 (mutated)
```

#### 7. Ana Döngü

```python
num_generations = 5

for generation in range(num_generations):
    print("\nGeneration:", generation)
    
    # 1. Fitness hesapla
    fitness = calc_pop_fitness(equation_inputs, new_population)
    
    # 2. Parent'ları seç
    parents = select_mating_pool(new_population, fitness, num_parents_mating)
    
    # 3. Crossover
    offspring_size = (sol_per_population - parents.shape[0], num_weights)
    offspring_crossover = crossover(parents, offspring_size)
    
    # 4. Mutation
    offspring_mutation = mutation(offspring_crossover)
    
    # 5. Yeni nesil
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
    
    # 6. En iyi fitness'ı göster
    best_val = np.max(np.sum(new_population * np.array(equation_inputs), axis=1))
    print("Best result:", best_val)
```

**Nesil Evrimi Örneği:**
```
Generation 0: Best fitness = 32.5
Generation 1: Best fitness = 45.8  (iyileşti)
Generation 2: Best fitness = 51.2  (iyileşti)
Generation 3: Best fitness = 51.2  (sabit)
Generation 4: Best fitness = 53.7  (iyileşti)
```

#### 8. Final Sonuç

```python
fitness = calc_pop_fitness(equation_inputs, new_population)
best_idx = int(np.argmax(fitness))
print("\nBest solution (weights):", new_population[best_idx, :])
print("Best solution fitness:", fitness[best_idx])
```

**Örnek Çıktı:**
```
Best solution (weights): [ 3.89, -3.12,  3.76,  3.95, -3.85, -3.91]
Best solution fitness: 124.38
```

**Yorumlama:**
- w₁ = 3.89: 4 × 3.89 = 15.56 ✓
- w₂ = -3.12: -2 × (-3.12) = 6.24 ✓ (eksi × eksi = artı)
- w₃ = 3.76: 3.5 × 3.76 = 13.16 ✓
- w₄ = 3.95: 5 × 3.95 = 19.75 ✓
- w₅ = -3.85: -11 × (-3.85) = 42.35 ✓ (eksi × eksi = artı)
- w₆ = -3.91: -4.7 × (-3.91) = 18.38 ✓ (eksi × eksi = artı)

**Optimal Strateji:**
- Pozitif coefficient → Pozitif weight
- Negatif coefficient → Negatif weight
- Büyük mutlak değerler → Büyük katkı

## Gerekli Kütüphaneler

```python
import numpy as np
```

## Genetic Algorithm Parametreleri

| Parametre | Değer | Etki |
|-----------|-------|------|
| **Population Size** | 8 | Küçük→Hızlı ama az çeşitlilik, Büyük→Yavaş ama iyi exploration |
| **Num Parents** | 4 | En iyi %50'sini seç |
| **Crossover Point** | 3 (ortası) | Tek nokta, basit ve etkili |
| **Mutation Rate** | Her offspring'de 1 gen | Küçük mutation, stability için |
| **Num Generations** | 5 | Daha fazla→Daha iyi sonuç (ama diminishing returns) |

## Genetik Algoritma Avantajları

✓ **Global optimum bulma şansı**: Local optimum'a takılmaz
✓ **Türev gerektirmez**: Black-box optimizasyon
✓ **Paralelize edilebilir**: Fitness hesaplama bağımsız
✓ **Discrete ve continuous değişkenler**: Her ikisi için de çalışır
✓ **Multi-objective optimization**: Birden fazla hedef

## Dezavantajlar

✗ **Yavaş**: Çok fitness evaluation gerekir
✗ **Hyperparameter tuning**: Population size, mutation rate, vb. ayar gerekir
✗ **Garanti yok**: Optimal çözüm bulacağının garantisi yok
✗ **Premature convergence**: Erken yakınsama riski

## Ne Zaman Kullanılır?

- **NP-hard problemler**: Traveling salesman, knapsack
- **Continuous optimization**: Neural network weight tuning
- **Scheduling**: İş çizelgeleme, rota optimizasyonu
- **Design optimization**: Mühendislik tasarımı
- **Feature selection**: ML'de feature seçimi
- **Game AI**: Oyun stratejileri

## Alternatif Algoritma Karşılaştırması

| Algoritma | Hız | Global Optimum | Türev Gerekir mi? |
|-----------|-----|----------------|-------------------|
| **Gradient Descent** | Hızlı | Hayır (local) | Evet |
| **Genetic Algorithm** | Yavaş | Muhtemel | Hayır |
| **Simulated Annealing** | Orta | Muhtemel | Hayır |
| **Particle Swarm** | Orta | Muhtemel | Hayır |
| **Random Search** | Çok Yavaş | Şanslıysa | Hayır |

## İleri Seviye Teknikler

### Adaptive Mutation Rate
```python
mutation_rate = 1.0 / (generation + 1)  # Nesil arttıkça azal
```

### Tournament Selection
```python
# Rastgele k birey seç, en iyisini al
tournament_size = 3
```

### Multi-Point Crossover
```python
# 2 veya daha fazla crossover noktası
crossover_points = [2, 4]
```

### Niching
- Farklı optimum'ları bulmak için popülasyonu kümelere ayır

## Notlar

- Fitness fonksiyonu, problem-specific'tir
- Mutation rate çok yüksek → Rastgele arama
- Mutation rate çok düşük → Premature convergence
- Elitism, en iyi çözümü kaybetmeyi engeller
- Crossover, parent'ların iyi özelliklerini birleştirir
- 5 nesil genelde yetersizdir, 50-100+ nesil önerilir
- Bu kod, linear equation için basit bir örnektir
