# Pharmaceutical Drug Sales Forecasting

A comprehensive Python solution for forecasting monthly sales volumes for pharmaceutical drugs post-generic entry.

## Overview

This project addresses the challenge of predicting pharmaceutical drug sales after generic alternatives enter the market. It implements two forecasting scenarios:

1. **Scenario 1**: Forecast 24 months of post-generic entry sales using only pre-entry data
2. **Scenario 2**: Forecast months 6-23 post-generic entry using the first 6 months of post-entry data

## Features

- **Data Preprocessing**: Load, clean, and prepare pharmaceutical sales data
- **Exploratory Data Analysis (EDA)**: Comprehensive visualizations of volume trends, erosion patterns, and drug characteristics
- **Feature Engineering**: Erosion bucket derivation, lag features, time-based features, and aggregated metrics
- **Model Building**: Ensemble models (Random Forest, Gradient Boosting, Ridge, ElasticNet), ARIMA, and LSTM-based forecasters
- **Evaluation**: Absolute and normalized error metrics, MAPE, RMSE, R², and accumulated error calculations
- **Insights Generation**: Business insights on erosion patterns, therapeutic area analysis, and drug type comparisons

## Project Structure

```
df_generics_test1/
├── data/                          # Data directory
│   ├── df_volume.csv              # Volume data
│   ├── df_generics.csv            # Generics information
│   └── df_medicine_info.csv       # Medicine characteristics
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data loading and cleaning
│   ├── eda.py                     # Exploratory data analysis
│   ├── feature_engineering.py     # Feature engineering
│   ├── models.py                  # Forecasting models
│   ├── evaluation.py              # Model evaluation
│   └── insights.py                # Business insights generation
├── outputs/                       # Generated outputs and visualizations
├── notebooks/                     # Jupyter notebooks (optional)
├── main.py                        # Main execution script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pichain23/df_generics_test1.git
cd df_generics_test1
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Complete Pipeline

```bash
python main.py
```

### Command Line Options

```bash
python main.py --data-path data/ --output-path outputs/ --scenario 1
```

- `--data-path`: Path to data directory (default: `data/`)
- `--output-path`: Path for output files (default: `outputs/`)
- `--skip-eda`: Skip exploratory data analysis
- `--scenario`: Run specific scenario (1 or 2). If not specified, runs both.

### Using Individual Modules

```python
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import Scenario1Model, Scenario2Model
from src.evaluation import ForecastEvaluator

# Load and preprocess data
preprocessor = DataPreprocessor(data_path='data/')
df = preprocessor.preprocess_pipeline()

# Engineer features
fe = FeatureEngineer()
df_engineered = fe.run_feature_engineering_pipeline(df)

# Train Scenario 1 model
model = Scenario1Model(model_type='ensemble')
model.fit(df_engineered)
predictions = model.predict(df_engineered, drug_id=1, n_months=24)

# Evaluate
evaluator = ForecastEvaluator()
metrics = evaluator.evaluate_forecast(actual, predicted, pre_entry_avg)
```

## Data Format

### df_volume.csv
| Column | Description |
|--------|-------------|
| drug_id | Unique drug identifier |
| brand_name | Drug brand name |
| country | Country of sale |
| months_postgx | Months relative to generic entry (negative = pre-entry) |
| volume | Sales volume |

### df_generics.csv
| Column | Description |
|--------|-------------|
| drug_id | Unique drug identifier |
| months_postgx | Months post-generic entry |
| n_gxs | Number of generic entrants |
| gx_market_share | Generic market share |

### df_medicine_info.csv
| Column | Description |
|--------|-------------|
| drug_id | Unique drug identifier |
| brand_name | Drug brand name |
| therapeutic_area | Therapeutic classification |
| hospital_rate | Hospital channel percentage |
| biological | Is biological drug (0/1) |
| small_molecule | Is small molecule drug (0/1) |

## Evaluation Metrics

The models are evaluated using:

1. **Absolute Monthly Error**: |Actual - Predicted|
2. **Normalized Error**: Absolute error / Pre-entry average volume
3. **Accumulated Error**: Sum of normalized errors over forecast horizon
4. **MAPE**: Mean Absolute Percentage Error
5. **RMSE**: Root Mean Squared Error
6. **R²**: Coefficient of determination

## Key Insights

The analysis generates insights on:

- High-erosion drugs and their characteristics
- Erosion patterns by therapeutic area
- Biological vs. small molecule drug erosion comparison
- Hospital rate impact on erosion
- Generic entrant effects on brand erosion
- Bucket 1 (low erosion) drug analysis

## Outputs

The pipeline generates:

- **Predictions**: CSV files with forecasted volumes for both scenarios
- **Visualizations**: PNG files with volume trends, erosion analysis, and evaluation plots
- **Reports**: Text summaries of business insights
- **Evaluation Results**: CSV files with model performance metrics

## Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- statsmodels >= 0.13.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- tensorflow >= 2.10.0 (optional, for LSTM models)

## License

MIT License