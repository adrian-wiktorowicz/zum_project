# Projekt Zastosowania Uczenia Maszynowego

## 1. Informacje ogólne

| | |
|---|---|
| **Nazwa projektu:** | Klasyfikacja podgatunków muzyki elektronicznej (EDM) |
| **Autor:** | Adrian Wiktorowicz |
| **Kierunek, rok i tryb studiów:** | Informatyka, semestr 3, internetowe |
| **Data oddania projektu:** | 18.01.2026 |

---

## 2. Opis projektu

Celem projektu jest **klasyfikacja 10-sekundowych fragmentów audio** do jednej z czterech klas podgatunków muzyki elektronicznej:

- 🏠 **House** – Four-on-the-floor beats, soulful vocals
- 🔊 **Techno** – Dark, repetitive, industrial sounds  
- 🌀 **Trance** – Euphoric melodies, build-ups and breakdowns
- 🥁 **Drum and Bass** – Fast breakbeats (160-180 BPM), heavy bass

Projekt wykorzystuje różne podejścia do uczenia maszynowego: **klasyczne ML (Random Forest)**, **sieci neuronowe zbudowane od zera (CNN)** oraz **modele transformerowe (AST)** do klasyfikacji na podstawie spektrogramów mel. Może być użyteczny w automatycznym tagowaniu muzyki, systemach rekomendacji oraz analizie trendów muzycznych.

---

## 3. Dane

| | |
|---|---|
| **Źródło danych:** | MTG-Jamendo Dataset + Jamendo API |
| **Link do metadanych:** | [MTG-Jamendo Dataset](https://github.com/MTG/mtg-jamendo-dataset) |
| **Link do audio API:** | [Jamendo API](https://developer.jamendo.com) |

### Opis danych:

| Parametr | Wartość |
|----------|---------|
| Liczba próbek | 2517 |
| Liczba klas | 4 (house, techno, trance, drum_and_bass) |
| Format danych | MP3 preview (96 kbps, 10s fragmenty) → Mel spectrogramy |
| Podział danych | 70% train / 15% val / 15% test |
| Licencja metadanych | CC BY 4.0 |
| Licencja audio | CC BY-NC-SA 3.0 |

> [!NOTE]
> **Uwaga dotycząca danych:**
> Pełny zbiór audio **nie jest wrzucony do repozytorium** ze względu na rozmiar i licencję. Audio pobierane jest dynamicznie z Jamendo API w notebooku `0_Data_Acquisition.ipynb`.

### Przetwarzanie danych:
- Normalizacja tagów (lowercase, strip)
- Mapowanie synonimów (np. "dnb" → "drum_and_bass")
- Zachowanie tylko tracków z **dokładnie 1** pasującym tagiem (single-label)
- Balansowanie: max 1000 próbek per klasa
- Artist-disjoint split (ten sam artysta nie występuje w różnych splitach)

---

## 4. Cel projektu

### Cel biznesowy/badawczy:
- **Co robi model?** Automatycznie klasyfikuje fragmenty audio do jednego z czterech podgatunków EDM.
- **Jakie pytanie odpowiada?** Do jakiego podgatunku muzyki elektronicznej należy dany utwór?
- **Jakie wnioski można wyciągnąć?**
  - Porównanie skuteczności różnych architektur (klasyczne ML vs. CNN vs. Transformery)
  - Analiza trudności rozróżniania podobnych podgatunków (np. house vs. techno)
  - Ocena przydatności transfer learningu w audio classification

---

## 5. Struktura projektu

Projekt składa się z pięciu głównych etapów, każdy w osobnym notatniku `.ipynb`:

| Etap | Nazwa pliku | Opis |
|------|-------------|------|
| 0 | `0_Data_Acquisition.ipynb` | Pobieranie metadanych i audio z Jamendo API |
| 1 | `1_EDA.ipynb` | Eksploracyjna analiza danych, wizualizacje, wnioski |
| 2 | `2_Preprocessing_Features.ipynb` | Przetwarzanie audio, ekstrakcja cech, spektrogramy mel |
| 3 | `3_Models_Training.ipynb` | Trening modeli: Random Forest, CNN, AST |
| 4 | `4_Evaluation.ipynb` | Ewaluacja, porównanie modeli, wizualizacje wyników |

---

## 6. Modele

Projekt obejmuje **trzy różne podejścia** do modelowania danych:

### 6.1 Model klasyczny ML – Random Forest

| | |
|---|---|
| **Algorytm:** | Random Forest Classifier |
| **Liczba drzew:** | 200 |
| **Max głębokość:** | 20 |
| **Min samples split:** | 5 |
| **Cechy wejściowe:** | Statystyki ze spektrogramu mel (mean, std, max, min per mel bin) |

**Krótki opis działania:**  
Model używa statystycznych cech wyekstrahowanych z log-mel spektrogramów. Dla każdego z 128 mel bins obliczane są statystyki (średnia, odchylenie, min, max), tworząc wektor cech 512-wymiarowy. Random Forest podejmuje decyzję na podstawie głosowania 200 drzew.

**Wyniki:**

| Metryka | Wartość |
|---------|---------|
| Accuracy | 0.3110 (31.10%) |
| Macro F1 | 0.2471 |
| Macro Precision | 0.2406 |
| Macro Recall | 0.2881 |

---

### 6.2 Sieć neuronowa zbudowana od zera – Simple CNN

| | |
|---|---|
| **Architektura:** | 4-block Convolutional Neural Network |
| **Liczba parametrów:** | ~200K |

**Struktura warstw:**

| Blok | Operacje | Channels | Dropout |
|------|----------|----------|---------|
| 1 | Conv2d → BatchNorm → ReLU → MaxPool(2) | 1 → 16 | 0.1 |
| 2 | Conv2d → BatchNorm → ReLU → MaxPool(2) | 16 → 32 | 0.2 |
| 3 | Conv2d → BatchNorm → ReLU → MaxPool(2) | 32 → 64 | 0.3 |
| 4 | Conv2d → BatchNorm → ReLU → MaxPool(2) | 64 → 128 | 0.4 |
| FC | Adaptive Avg Pool → Linear → Dropout(0.5) → Linear | 128 → 64 → 4 | 0.5 |

**Funkcje aktywacji:** ReLU  
**Optymalizator:** AdamW (lr=1e-3, weight_decay=1e-4)  
**Epoki:** 30 (early stopping patience=7)  
**Augmentacja:** SpecAugment (freq_mask=10, time_mask=20)

**Wyniki:**

| Metryka | Wartość |
|---------|---------|
| Accuracy | 0.3307 (33.07%) |
| Macro F1 | 0.2668 |
| Macro Precision | 0.2573 |
| Macro Recall | 0.3016 |

---

### 6.3 Model transformerowy (fine-tuning) – Audio Spectrogram Transformer (AST)

| | |
|---|---|
| **Nazwa modelu:** | MIT/ast-finetuned-audioset-10-10-0.4593 |
| **Biblioteka:** | HuggingFace Transformers |
| **Liczba parametrów:** | ~87M |
| **Strategia fine-tuningu:** | Zamrożenie encodera + odmrożenie ostatnich 2 bloków |

**Zakres dostosowania:**  
- Nowa warstwa klasyfikacji (4 klasy zamiast 527 AudioSet)
- Fine-tuning ostatnich 2 bloków transformera
- Optymalizator: AdamW (lr=1e-5)
- Warmup: 100 kroków
- Epoki: 10

**Wyniki:**

| Metryka | Wartość |
|---------|---------|
| Accuracy | **0.5276 (52.76%)** |
| Macro F1 | **0.5563** |
| Macro Precision | 0.5772 |
| Macro Recall | 0.5427 |

---

## 7. Ewaluacja

### Użyte metryki:
- **Accuracy** – ogólna poprawność klasyfikacji
- **Macro F1** – zbalansowana średnia F1 per klasa (główna metryka)
- **Macro Precision** – średnia precyzja per klasa
- **Macro Recall** – średnia czułość per klasa

### Porównanie modeli:

| Model | Accuracy | Macro F1 | Uwagi |
|-------|----------|----------|-------|
| Random Forest | 0.3110 | 0.2471 | Najsłabszy – cechy statystyczne niewystarczające |
| CNN | 0.3307 | 0.2668 | Marginalna poprawa względem RF |
| **AST** | **0.5276** | **0.5563** | **Najlepszy – transfer learning z AudioSet** |

### Wizualizacje (w folderze `results/`):

| Wizualizacja | Plik |
|--------------|------|
| Macierz pomyłek – wszystkie modele | `all_confusion_matrices.png` |
| Macierz pomyłek – Random Forest | `confusion_matrix_rf.png` |
| Macierz pomyłek – CNN | `confusion_matrix_cnn.png` |
| Macierz pomyłek – AST | `confusion_matrix_ast.png` |
| Krzywe uczenia – CNN | `cnn_learning_curves.png` |
| Krzywe uczenia – AST | `ast_learning_curves.png` |
| Porównanie modeli | `model_comparison.png` |
| F1 per klasa | `per_class_f1.png` |
| Analiza błędów | `error_analysis.png` |

---

## 8. Wnioski i podsumowanie

### Który model okazał się najlepszy i dlaczego?

**Audio Spectrogram Transformer (AST)** osiągnął najlepsze wyniki z Macro F1 = 0.5563, znacząco przewyższając modele klasyczne. Wynika to z:
- **Transfer learning** – model pretrenowany na AudioSet (2M próbek, 527 klas) posiada bogate reprezentacje audio
- **Architektura Transformer** – skuteczne modelowanie długozasięgowych zależności w spektrogramach
- **Fine-tuning** – dostosowanie do specyfiki podgatunków EDM

### Trudności podczas pracy:
1. **Nakładanie się podgatunków** – EDM subgenres mają wspólne cechy (4/4 beat, syntetyczne brzmienia)
2. **Single-label constraint** – wiele utworów pasuje do więcej niż jednej kategorii
3. **Jakość danych** – preview 96kbps MP3 zamiast pełnych utworów
4. **Rozmiar datasetu** – filtrowanie single-label znacząco zredukowało liczbę próbek

### Co można poprawić w przyszłości?
1. Multi-label classification zamiast single-label
2. Większy i bardziej zbalansowany dataset
3. Ensemble różnych modeli
4. Augmentacja audio (pitch shift, time stretch)
5. Dłuższe fragmenty audio (30s zamiast 10s)

### Potencjalne zastosowania:
- Automatyczne tagowanie utworów w serwisach streamingowych
- Systemy rekomendacji muzyki
- Analiza trendów w muzyce elektronicznej
- Asystent DJ-a do organizacji biblioteki muzycznej

---

## 9. Struktura repozytorium

```
zum_project/
│
├── data/
│   ├── raw/              # Surowe metadane TSV
│   ├── processed/        # Manifest CSV
│   └── audio/            # Pobrane MP3 (gitignored)
├── notebooks/
│   ├── 0_Data_Acquisition.ipynb
│   ├── 1_EDA.ipynb
│   ├── 2_Preprocessing_Features.ipynb
│   ├── 3_Models_Training.ipynb
│   └── 4_Evaluation.ipynb
├── src/
│   ├── config.py         # Konfiguracja i hiperparametry
│   ├── data_utils.py     # Pobieranie i przetwarzanie danych
│   ├── audio_utils.py    # Przetwarzanie audio
│   ├── dataset.py        # PyTorch Dataset
│   ├── models.py         # Architektury modeli (CNN, RF, AST)
│   ├── training.py       # Pętla treningowa
│   └── evaluation.py     # Metryki i wizualizacje
├── models/               # Checkpointy modeli (gitignored)
├── results/              # Wyniki i wykresy
├── requirements.txt
└── README.md
```

---

## 10. Technologia i biblioteki

| Kategoria | Technologie |
|-----------|-------------|
| **Język** | Python 3.10+ |
| **Deep Learning** | PyTorch, torchaudio |
| **Transformers** | HuggingFace Transformers (AST) |
| **Klasyczne ML** | scikit-learn (Random Forest) |
| **Przetwarzanie audio** | librosa, torchaudio |
| **Analiza danych** | NumPy, Pandas |
| **Wizualizacja** | Matplotlib, Seaborn, Plotly |

### Główne parametry przetwarzania audio:

| Parametr | Wartość |
|----------|---------|
| Sample rate | 22050 Hz |
| Clip duration | 10 sekund |
| Mel bins | 128 |
| FFT size | 1024 |
| Hop length | 256 |

---

## 11. Instalacja i uruchomienie

### 1. Klonowanie repozytorium

```bash
git clone <repo-url>
cd zum_project
```

### 2. Tworzenie środowiska wirtualnego

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalacja zależności

```bash
pip install -r requirements.txt
```


### 4. Uruchomienie notebooków

```bash
jupyter notebook notebooks/

```

---

## 12. Licencja projektu

| Element | Licencja |
|---------|----------|
| Kod projektu | MIT License |
| Metadane MTG-Jamendo | CC BY 4.0 |
| Audio Jamendo | CC BY-NC-SA 3.0 (tylko do celów niekomercyjnych) |
