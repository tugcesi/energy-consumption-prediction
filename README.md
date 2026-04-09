# ⚡ Energy Consumption Prediction

Bu proje, bir evin dakika bazlı elektrik tüketimi verilerini kullanarak **Global Aktif Güç** değerini tahmin eden bir **derin öğrenme (LSTM)** çalışmasıdır. UCI Machine Learning Repository'den alınan gerçek dünya verisi ile eğitilmiştir.

---

## 📁 Proje Yapısı

```
📦 energy-consumption-prediction
├── 📓 EnergyConsumptionPrediction.ipynb   # Ana analiz ve model notebook'u
├── 🗜️ housedold_power_consumption.zip     # Ham veri seti (sıkıştırılmış)
└── 📄 README.md
```

> ⚠️ Eğitilmiş model dosyası (`lstm_model.h5` / `.keras`) boyutu nedeniyle repoya eklenememiştir. Modeli yeniden üretmek için notebook'u çalıştırabilirsiniz.

---

## 📊 Veri Seti

| Özellik | Detay |
|---|---|
| **Kaynak** | UCI Machine Learning Repository |
| **Dosya** | `household_power_consumption.txt` |
| **Toplam Kayıt** | 2.075.259 (dakika bazlı) |
| **Temizleme Sonrası** | 2.049.280 satır |
| **Tarih Aralığı** | 16/12/2006 – 26/11/2010 |
| **Hedef Değişken** | `Global_active_power` (kW) |

### Sütunlar

| Sütun | Açıklama |
|---|---|
| `Global_active_power` | Küresel aktif güç (kilowatt) — **hedef** |
| `Global_reactive_power` | Küresel reaktif güç (kilowatt) |
| `Voltage` | Ortalama voltaj (volt) |
| `Global_intensity` | Ortalama akım yoğunluğu (amper) |
| `Sub_metering_1` | Mutfak ekipmanları alt sayacı |
| `Sub_metering_2` | Çamaşır/banyo ekipmanları alt sayacı |

---

## 🔬 Yapılan Analizler

- ✅ Keşifsel Veri Analizi (EDA)
- ✅ Günlük ve saatlik ortalama güç grafiği
- ✅ Korelasyon matrisi (heatmap)
- ✅ Eksik değer temizleme (25.979 satır)
- ✅ MinMaxScaler normalizasyonu
- ✅ Sequence oluşturma (lookback = 60 dakika)
- ✅ %80 / %20 Train-Test bölümü
- ✅ LSTM modeli eğitimi (5 epoch)
- ✅ Model değerlendirmesi (MSE, MAE)

---

## 🧠 Model Mimarisi

```
LSTM (50 unit, relu, return_sequences=True)
    ↓ Dropout(0.2)
LSTM (50 unit, relu)
    ↓ Dropout(0.2)
Dense(1)
```

| Parametre | Değer |
|---|---|
| **Optimizer** | Adam (lr=0.001) |
| **Loss** | MSE |
| **Metric** | MAE |
| **Epochs** | 5 |
| **Batch Size** | 32 |
| **Lookback** | 60 dakika |

---

## 🚀 Kurulum ve Kullanım

### Gereksinimler

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras
```

### Çalıştırma

```bash
git clone https://github.com/tugcesi/energy-consumption-prediction.git
cd energy-consumption-prediction

# Veri dosyasını çıkar
unzip housedold_power_consumption.zip

# Notebook'u çalıştır
jupyter notebook EnergyConsumptionPrediction.ipynb
```

---

## 🛠️ Kullanılan Teknolojiler

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-LSTM-red?logo=keras)
![Pandas](https://img.shields.io/badge/Pandas-library-green?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## 📄 Lisans

Bu proje [MIT License](LICENSE) ile lisanslanmıştır.