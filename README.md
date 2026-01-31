# Sleep Stage Prediction with Optuna + XGBoost

Predicting sleep stages (Wake, N1, N2, N3, REM) from heart rate data using XGBoost with Optuna hyperparameter optimization.

## Dataset

This project uses the [Motion and Heart Rate from a Wrist-Worn Wearable and Labeled Sleep from Polysomnography](https://physionet.org/content/sleep-accel/1.0.0/) dataset from PhysioNet.

- **31 subjects** with overnight sleep recordings
- **Heart rate** from wrist-worn PPG sensor
- **Sleep stages** labeled via polysomnography (gold standard)

## Quick Start (Google Colab)

### Baseline: HR Only
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Baglecake/HR_sleep/blob/main/sleep_stage_prediction_colab.ipynb)

Uses heart rate data only (~4MB download).

### Ablation Study: HR + Motion + Steps
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Baglecake/HR_sleep/blob/main/sleep_stage_ablation_study.ipynb)

Toggle data modalities to compare performance. ⚠️ Motion data is ~2GB.

| Config | Expected Accuracy |
|--------|-------------------|
| HR only | ~30-40% |
| HR + Motion | ~55-70% |
| HR + Motion + Steps | ~56-71% |

## Local Installation

```bash
# Clone the repo
git clone https://github.com/Baglecake/HR_sleep.git
cd HR_sleep

# Install dependencies
pip install -r requirements.txt

# Download data from PhysioNet (heart_rate and labels folders only)
# Place in the same directory as the scripts

# Run
python sleep_stage_optuna_xgboost.py
```

## Features Extracted

HRV-like features computed from heart rate in 30-second epochs:

| Feature | Description |
|---------|-------------|
| `hr_mean` | Mean heart rate |
| `hr_std` | Standard deviation |
| `hr_min`, `hr_max`, `hr_range` | Range metrics |
| `hr_median` | Median heart rate |
| `hr_cv` | Coefficient of variation |
| `hr_iqr` | Interquartile range |
| `hr_skew` | Skewness |
| `hr_rmssd` | Root Mean Square of Successive Differences |
| `hr_pnn50` | % of successive differences > 5 BPM |
| `hr_slope` | Linear trend within epoch |
| `hr_count` | Number of samples in epoch |

## Methodology

### Hyperparameter Optimization
- **Optuna** with TPE sampler for Bayesian optimization
- **Median pruner** for early stopping of unpromising trials
- 100 trials (configurable)

### Cross-Validation
- **GroupKFold** to prevent subject leakage (critical for physiological data)
- 5-fold cross-validation

### Class Imbalance
- Inverse frequency sample weights
- Macro F1-score as optimization metric

### XGBoost Search Space
```python
{
    'booster': ['gbtree', 'dart'],
    'lambda': [1e-8, 10.0],      # L2 regularization
    'alpha': [1e-8, 10.0],       # L1 regularization
    'max_depth': [3, 10],
    'learning_rate': [1e-3, 0.3],
    'n_estimators': [50, 500],
    'min_child_weight': [1, 10],
    'subsample': [0.5, 1.0],
    'colsample_bytree': [0.5, 1.0],
    'gamma': [1e-8, 1.0]
}
```

## Output Files

After running, you'll get:
- `best_params.csv` - Optimal hyperparameters
- `sleep_stage_model.json` - Trained XGBoost model
- `optimization_history.csv` - All trial results

## Files

```
├── sleep_stage_optuna_xgboost.py       # Main Python script (HR only)
├── sleep_stage_prediction_colab.ipynb  # Colab notebook (HR only)
├── sleep_stage_ablation_study.ipynb    # Ablation study (HR + Motion + Steps)
├── requirements.txt                    # Dependencies
├── README.md                           # This file
└── .gitignore                          # Excludes data files
```

## Requirements

- Python 3.8+
- numpy
- pandas
- scikit-learn
- xgboost
- optuna

## License

The dataset is from PhysioNet and subject to their [data use agreement](https://physionet.org/content/sleep-accel/1.0.0/).

## References

- Walch, O. (2019). Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography (version 1.0.0). PhysioNet. https://doi.org/10.13026/hmhs-py35
