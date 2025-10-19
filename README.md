# Power System Availability Prediction & Energy Allocation

A machine learning system for predicting power system availability and optimizing energy allocation across different feeder types using deep learning models (BiLSTM, LSTM, GRU) and advanced hyperparameter optimization algorithms.

## 🎯 Project Overview

This project addresses the challenge of power distribution management by:
- **Predicting availability hours** for 33KV power feeders using historical data
- **Optimizing energy allocation** across feeder types (Residential, Commercial, Industrial, Healthcare)
- **Comparing multiple deep learning models** with various hyperparameter optimization strategies (Dragonfly, Hyperband, Optuna)
- **Providing real-time insights** via an interactive Streamlit web application

## 🏗️ Project Structure

```
power-system-availability-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── cleaned_data/
│   ├── agbara_data.csv
│   ├── data_stat.csv
│   └── new_data.csv
├── data/
├── model_metrics/
│   ├── best_model_metrics_sorted.csv
│   ├── best_model_metrics.csv
│   ├── BiLSTM_model.keras
│   └── model_metrics_comparison.csv
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── data_clean_and_analysis.ipynb
│   └── model_train.ipynb
└── tuning_params/
    ├── bilstm_tuning/
    ├── gru_tuning/
    └── lstm_tuning/
```

## 🚀 Features

### 🔮 Prediction Capabilities
- **Availability Forecasting**: Predict daily availability hours for power feeders
- **Multi-Model Support**: BiLSTM, LSTM, GRU, and traditional ML models
- **Advanced Hyperparameter Tuning**: Dragonfly (Bayesian, Random, Direct, PDOO), Hyperband, Optuna

### ⚡ Energy Allocation
- **Time-Window Based Allocation**: Different allocation strategies for different hours
- **Priority-Based Distribution**: Healthcare feeders get priority (40% allocation)
- **Proportional Allocation**: Remaining energy distributed based on consumption patterns

### 📊 Interactive Dashboard
- **Real-time Predictions**: Enter supply values and get instant availability predictions
- **Visual Analytics**: Bar charts, pie charts, and line plots for allocation insights
- **Feeder Selection**: Choose specific feeders for targeted analysis

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/Giwa-ibrahim/power-system-availability.git
cd power-system-availability
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify data files**
Ensure your data files are in the `data/` directory:
- 33KV Daily Availability (2019-2021).xlsx
- 33KV Daily Consumption (2019-2021).xlsx

## 🎮 Usage

### Running the Web Application

```bash
cd src
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Data Processing & Model Training

1. **Data Cleaning & Analysis**
```bash
jupyter notebook src/data_clean_and_analysis.ipynb
```

2. **Model Training & Evaluation**
```bash
jupyter notebook src/model_train.ipynb
```

## 🧠 Machine Learning Models & Optimization

### Model Architectures

#### BiLSTM (Best Performing)
```python
def build_bilstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
        Bidirectional(LSTM(32)),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    return model
```

#### LSTM & GRU
- Standard LSTM/GRU layers with dropout regularization
- Hyperparameter tuning using Keras Tuner, Dragonfly, and Optuna

### Hyperparameter Optimization Algorithms

- **Dragonfly**: Bayesian, Random, Direct, PDOO methods
- **Hyperband**: Successive halving and early stopping
- **Optuna**: Tree-structured Parzen Estimator (TPE)
- **Grid/Random Search**: Traditional methods

### Performance Metrics

Models are evaluated using:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Square Error)
- **R² Score** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error, with robust calculation to avoid division by zero issues)

## 📈 Energy Allocation Strategy

### Time-Based Windows
```python
time_windows = [
    (0, 5, ["Healthcare", "Residential"]),
    (5, 9, ["Healthcare", "Residential", "Commercial"]),
    (9, 12, ["Healthcare", "Industrial", "Commercial"]),
    (12, 15, ["Healthcare", "Industrial"]),
    (15, 18, ["Healthcare", "Industrial", "Commercial"]),
    (18, 23, ["Healthcare", "Residential", "Commercial"]),
    (23, 24, ["Healthcare", "Residential"]),
]
```

### Allocation Rules
1. **Healthcare Priority**: Always receives 40% when active
2. **Proportional Distribution**: Remaining 60% distributed based on historical consumption
3. **Time-Sensitive**: Different feeder combinations for different hours

## 📊 Data Features

### Input Features
- `feeder_id`
- `consumption_mwh`
- `feeder_type`
- `day_of_week`
- `month`
- `is_weekend`
- `lag1_avail`

### Target Variable
- `availability_hrs`: Daily availability hours (1-24)

## 🔧 Configuration

### Model Parameters
Key hyperparameters for BiLSTM model:
- **Units**: 128 (first layer), 32 (second layer)
- **Dropout Rate**: 0.3
- **Learning Rate**: 0.0001325
- **Batch Size**: 32
- **Epochs**: 50

### Logging
Application logs are saved to [`app.log`](src/app.py) with INFO level logging.

## 📝 Dependencies

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
tensorflow>=2.13.0
scikit-learn>=1.3.0
keras-tuner>=1.4.0
xgboost>=1.7.0
dragonfly-opt>=0.1.6
optuna>=3.0.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dr. Amole** - Project Supervisor
- **Bells University** - Research Support
- **Power Distribution Companies** - Data Provision
- **TensorFlow/Keras Team** - Deep Learning Framework
- **Streamlit Team** - Web Application Framework

## 📞 Contact

**Project Team**: Bells COLENG Research Group  
**Supervisor**: Dr. Amole  
**Institution**: Bells University of Technology

---

**Note**: This project is part of ongoing research in "Optimization of Power System Availability" at Bells University College of Engineering.