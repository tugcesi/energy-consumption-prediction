# Energy Consumption Prediction

A deep learning project that uses LSTM (Long Short-Term Memory) neural networks to predict household energy consumption based on historical power usage data.

## 📌 Project Description

This project analyzes the UCI Household Power Consumption dataset and builds an LSTM model to forecast global active power usage. It includes exploratory data analysis (EDA), feature engineering, data preprocessing, model training, and evaluation steps.

## 📊 Dataset

The dataset used is the **Individual Household Electric Power Consumption** dataset, which contains measurements of electric power consumption in one household over a period of approximately 4 years (2006–2010).

- **File:** `housedold_power_consumption.zip` (extract to get `household_power_consumption.txt`)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)
- **Records:** ~2 million minute-level measurements

### Features

| Feature | Description |
|---|---|
| `Date` | Date in format dd/mm/yyyy |
| `Time` | Time in format hh:mm:ss |
| `Global_active_power` | Household global minute-averaged active power (kW) |
| `Global_reactive_power` | Household global minute-averaged reactive power (kW) |
| `Voltage` | Minute-averaged voltage (V) |
| `Global_intensity` | Household global minute-averaged current intensity (A) |
| `Sub_metering_1` | Energy sub-metering No. 1 (Wh) – kitchen |
| `Sub_metering_2` | Energy sub-metering No. 2 (Wh) – laundry room |
| `Sub_metering_3` | Energy sub-metering No. 3 (Wh) – climate control |

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tugcesi/energy-consumption-prediction.git
   cd energy-consumption-prediction
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Extract the dataset:**
   ```bash
   unzip housedold_power_consumption.zip
   ```

## 📖 Usage

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `EnergyConsumptionPrediction.ipynb` in your browser.

3. Run cells sequentially from top to bottom.

### Notebook Sections

- **Data Loading** – Loading and inspecting the raw dataset
- **EDA (Exploratory Data Analysis)** – Visualizing daily/hourly patterns and correlation analysis
- **Feature Engineering** – Datetime parsing, data cleaning, and feature selection
- **Model Building** – LSTM model construction with Dropout layers
- **Training & Evaluation** – Model training and performance metrics (MSE/RMSE)

## 🧰 Requirements

See [requirements.txt](requirements.txt) for the full list of dependencies. Main packages:

- `pandas` – Data manipulation
- `numpy` – Numerical operations
- `matplotlib` / `seaborn` – Data visualization
- `scikit-learn` – Data preprocessing and metrics
- `tensorflow` / `keras` – LSTM model building

## 📁 Project Structure

```
energy-consumption-prediction/
├── EnergyConsumptionPrediction.ipynb   # Main analysis notebook
├── housedold_power_consumption.zip     # Dataset (zipped)
├── requirements.txt                    # Python dependencies
├── CONTRIBUTING.md                     # Contribution guidelines
├── LICENSE                             # MIT License
└── README.md                           # Project documentation
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 👤 Author

**Tugce Basyigit**
- GitHub: [@tugcesi](https://github.com/tugcesi)
