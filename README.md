# Power System Availability Prediction & Energy Allocation

Machine learning system for predicting power availability and optimizing energy allocation across feeder types using BiLSTM deep learning with advanced hyperparameter optimization.

## 🎯 Overview

- **Predict** daily availability hours for 33KV power feeders
- **Allocate** energy optimally across feeder types (Residential, Commercial, Industrial, Healthcare)
- **Optimize** model performance using Dragonfly, Hyperband, and Optuna algorithms
- **Visualize** predictions and allocation via interactive Streamlit dashboard

## 🏗️ Project Structure

```
├── app.py                           # Streamlit application (entry point)
├── utils.py                         # Prediction & allocation utilities
├── requirements.txt                 # Python dependencies
├── cleaned_data/
│   └── processed_data.csv          # Processed dataset with features
├── model_metrics/
│   └── BiLSTM Hyperband.keras      # Trained BiLSTM model
└── src/
    ├── data_clean_and_analysis.ipynb
    └── model_train best.ipynb      # Model training & optimization
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Giwa-ibrahim/power-system-availability.git
cd power-system-availability

# Install dependencies (Python 3.8+)
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

Access at `http://localhost:8501`

**Usage:**
1. Enter total energy supply (MWh)
2. Select feeder from dropdown
3. Choose prediction date
4. Click "Predict & Allocate"

## 🧠 Model Architecture

### BiLSTM (Best Model)
- **Input**: 7 timesteps × 7 features
- **Architecture**: Bidirectional LSTM layers with dropout
- **Output**: Predicted availability hours (1-24)

### Features Used
- `feeder_name` (encoded)
- `district` (encoded)
- `consumption_mwh`
- `lag1_avail`, `lag2_avail`, `lag3_avail`
- `yesterday_full`

### Hyperparameter Optimization
- **Dragonfly**: Bayesian optimization
- **Hyperband**: Successive halving
- **Optuna**: Tree-structured Parzen Estimator

### Evaluation Metrics
MAE, MSE, RMSE, R², MAPE

## ⚡ Energy Allocation Strategy

### Time-Based Priority Windows
```python
time_windows = [
    (0-5h: Healthcare, Residential),
    (5-9h: Healthcare, Residential, Commercial),
    (9-12h: Healthcare, Industrial, Commercial),
    (12-15h: Healthcare, Industrial),
    (15-18h: Healthcare, Industrial, Commercial),
    (18-23h: Healthcare, Residential, Commercial),
    (23-24h: Healthcare, Residential)
]
```

### Allocation Rules
1. **Healthcare Priority**: 40% allocation when active
2. **Proportional Distribution**: Remaining 60% based on consumption patterns
3. **Time-Sensitive**: Feeder combinations vary by hour

## 📊 Key Dependencies

```txt
streamlit>=1.28.0
tensorflow>=2.13.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.6.0
keras-tuner>=1.4.0
optuna>=3.0.0
dragonfly-opt>=0.1.6
```

## 📝 Model Training

To retrain or experiment with models:

```bash
jupyter notebook src/model_train\ best.ipynb
```

## 🙏 Acknowledgments

**Supervisor**: Dr. Amole  
**Institution**: Bells University of Technology - College of Engineering  
**Project**: Optimization of Power System Availability

---

*Research project conducted at Bells COLENG, 2026*