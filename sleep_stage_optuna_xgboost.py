"""
Sleep Stage Prediction using Heart Rate Data with Optuna + XGBoost

This script uses only heart rate data and labeled sleep stages to predict
sleep stages (Wake, N1, N2, N3, REM) using XGBoost with Optuna hyperparameter tuning.

Dataset: Motion and Heart Rate from a Wrist-Worn Wearable and Labeled Sleep from Polysomnography
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path(__file__).parent
HEART_RATE_DIR = DATA_DIR / "heart_rate"
LABELS_DIR = DATA_DIR / "labels"

# Sleep stage mapping (from dataset)
SLEEP_STAGE_NAMES = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    5: 'REM'
}

# Epoch duration in seconds (standard PSG)
EPOCH_DURATION = 30


def load_heart_rate_data(subject_id: str) -> pd.DataFrame:
    """Load heart rate data for a subject."""
    hr_file = HEART_RATE_DIR / f"{subject_id}_heartrate.txt"

    if not hr_file.exists():
        raise FileNotFoundError(f"Heart rate file not found: {hr_file}")

    # Heart rate files are comma-separated with columns: timestamp, heart_rate
    df = pd.read_csv(hr_file, header=None, names=['timestamp', 'heart_rate'])
    df['subject_id'] = subject_id
    return df


def load_sleep_labels(subject_id: str) -> pd.DataFrame:
    """Load labeled sleep data for a subject."""
    label_file = LABELS_DIR / f"{subject_id}_labeled_sleep.txt"

    if not label_file.exists():
        raise FileNotFoundError(f"Labels file not found: {label_file}")

    # Labels are space-separated with columns: time_offset, sleep_stage
    df = pd.read_csv(label_file, sep=' ', header=None, names=['time_offset', 'sleep_stage'])
    df['subject_id'] = subject_id
    return df


def get_subject_ids() -> List[str]:
    """Get list of all subject IDs from the heart rate directory."""
    hr_files = glob.glob(str(HEART_RATE_DIR / "*_heartrate.txt"))
    subject_ids = [Path(f).stem.replace('_heartrate', '') for f in hr_files]
    return sorted(subject_ids)


def extract_hrv_features(hr_values: np.ndarray, timestamps: np.ndarray) -> Dict[str, float]:
    """
    Extract heart rate variability (HRV) features from heart rate data.

    Features extracted:
    - Time-domain: mean HR, std HR, min HR, max HR, range HR
    - HRV-like: RMSSD approximation, pNN50 approximation
    - Trend: slope of HR over the epoch
    """
    features = {}

    if len(hr_values) < 2:
        # Return NaN features if not enough data points
        return {
            'hr_mean': np.nan, 'hr_std': np.nan, 'hr_min': np.nan,
            'hr_max': np.nan, 'hr_range': np.nan, 'hr_median': np.nan,
            'hr_rmssd': np.nan, 'hr_pnn50': np.nan, 'hr_slope': np.nan,
            'hr_count': len(hr_values), 'hr_cv': np.nan,
            'hr_skew': np.nan, 'hr_iqr': np.nan
        }

    # Basic statistics
    features['hr_mean'] = np.mean(hr_values)
    features['hr_std'] = np.std(hr_values)
    features['hr_min'] = np.min(hr_values)
    features['hr_max'] = np.max(hr_values)
    features['hr_range'] = features['hr_max'] - features['hr_min']
    features['hr_median'] = np.median(hr_values)
    features['hr_count'] = len(hr_values)

    # Coefficient of variation
    if features['hr_mean'] > 0:
        features['hr_cv'] = features['hr_std'] / features['hr_mean']
    else:
        features['hr_cv'] = np.nan

    # IQR
    q75, q25 = np.percentile(hr_values, [75, 25])
    features['hr_iqr'] = q75 - q25

    # Skewness approximation
    if features['hr_std'] > 0:
        features['hr_skew'] = np.mean(((hr_values - features['hr_mean']) / features['hr_std']) ** 3)
    else:
        features['hr_skew'] = np.nan

    # HRV-like features (using HR differences as proxy for RR intervals)
    hr_diff = np.diff(hr_values)

    # RMSSD: Root Mean Square of Successive Differences
    if len(hr_diff) > 0:
        features['hr_rmssd'] = np.sqrt(np.mean(hr_diff ** 2))
    else:
        features['hr_rmssd'] = np.nan

    # pNN50: Percentage of successive differences > 50ms (adapted for HR)
    # Using threshold of 5 BPM as proxy
    if len(hr_diff) > 0:
        features['hr_pnn50'] = np.sum(np.abs(hr_diff) > 5) / len(hr_diff)
    else:
        features['hr_pnn50'] = np.nan

    # Slope (trend) of heart rate over the epoch
    if len(timestamps) >= 2 and len(hr_values) >= 2:
        try:
            slope, _ = np.polyfit(timestamps - timestamps[0], hr_values, 1)
            features['hr_slope'] = slope
        except:
            features['hr_slope'] = np.nan
    else:
        features['hr_slope'] = np.nan

    return features


def align_hr_to_epochs(
    hr_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    epoch_duration: int = EPOCH_DURATION
) -> pd.DataFrame:
    """
    Align heart rate data to sleep epoch labels.

    For each 30-second epoch in the labels, extract features from the
    heart rate data that falls within that epoch.
    """
    # Get the time range from labels
    label_start = labels_df['time_offset'].min()
    label_end = labels_df['time_offset'].max() + epoch_duration

    # Filter HR data to roughly match the label time range
    # Add some buffer for edge epochs
    hr_df_filtered = hr_df[
        (hr_df['timestamp'] >= label_start - epoch_duration) &
        (hr_df['timestamp'] <= label_end + epoch_duration)
    ].copy()

    features_list = []

    for _, row in labels_df.iterrows():
        epoch_start = row['time_offset']
        epoch_end = epoch_start + epoch_duration

        # Get HR samples within this epoch
        epoch_hr = hr_df_filtered[
            (hr_df_filtered['timestamp'] >= epoch_start) &
            (hr_df_filtered['timestamp'] < epoch_end)
        ]

        if len(epoch_hr) > 0:
            features = extract_hrv_features(
                epoch_hr['heart_rate'].values,
                epoch_hr['timestamp'].values
            )
        else:
            features = extract_hrv_features(np.array([]), np.array([]))

        features['time_offset'] = epoch_start
        features['sleep_stage'] = row['sleep_stage']
        features['subject_id'] = row['subject_id']

        features_list.append(features)

    return pd.DataFrame(features_list)


def prepare_dataset() -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and prepare the complete dataset from all subjects.
    """
    subject_ids = get_subject_ids()
    print(f"Found {len(subject_ids)} subjects")

    all_features = []

    for i, subject_id in enumerate(subject_ids):
        print(f"Processing subject {i+1}/{len(subject_ids)}: {subject_id}")

        try:
            # Load data
            hr_df = load_heart_rate_data(subject_id)
            labels_df = load_sleep_labels(subject_id)

            # Align and extract features
            features_df = align_hr_to_epochs(hr_df, labels_df)
            all_features.append(features_df)

            print(f"  - {len(features_df)} epochs extracted")

        except Exception as e:
            print(f"  - Error processing subject {subject_id}: {e}")
            continue

    if not all_features:
        raise ValueError("No data could be loaded from any subject")

    dataset = pd.concat(all_features, ignore_index=True)

    # Filter out invalid sleep stages (-1 and 4)
    valid_stages = [0, 1, 2, 3, 5]
    dataset = dataset[dataset['sleep_stage'].isin(valid_stages)].copy()

    # Drop rows with too many missing HR samples
    dataset = dataset[dataset['hr_count'] >= 2].copy()

    # Drop rows with NaN in critical features
    dataset = dataset.dropna(subset=['hr_mean', 'hr_std'])

    # =============================================================
    # ADD TEMPORAL CONTEXT FEATURES (Critical for sleep staging!)
    # =============================================================
    print("\nAdding temporal context features...")

    # Sort by subject and time (CRITICAL for lag/lead to work correctly)
    dataset = dataset.sort_values(['subject_id', 'time_offset']).reset_index(drop=True)

    # 1. Rolling averages (trend context) - 5 epochs = 2.5 minutes window
    dataset['hr_mean_roll_5'] = dataset.groupby('subject_id')['hr_mean'].transform(
        lambda x: x.rolling(window=5, center=True, min_periods=1).mean()
    )
    dataset['hr_std_roll_5'] = dataset.groupby('subject_id')['hr_std'].transform(
        lambda x: x.rolling(window=5, center=True, min_periods=1).mean()
    )

    # 2. Lag features (past context) - what happened before?
    for lag in [1, 2, 4]:  # 30s, 60s, 2min ago
        dataset[f'hr_mean_lag_{lag}'] = dataset.groupby('subject_id')['hr_mean'].shift(lag)
        dataset[f'hr_std_lag_{lag}'] = dataset.groupby('subject_id')['hr_std'].shift(lag)

    # 3. Lead features (future context) - what happens next? (valid for offline analysis)
    for lead in [1, 2]:  # 30s, 60s ahead
        dataset[f'hr_mean_lead_{lead}'] = dataset.groupby('subject_id')['hr_mean'].shift(-lead)

    # 4. Rate of change (is HR dropping or rising?)
    dataset['hr_diff_1'] = dataset['hr_mean'] - dataset.groupby('subject_id')['hr_mean'].shift(1)
    dataset['hr_diff_2'] = dataset['hr_mean'] - dataset.groupby('subject_id')['hr_mean'].shift(2)

    # 5. Variability change
    dataset['hr_std_diff_1'] = dataset['hr_std'] - dataset.groupby('subject_id')['hr_std'].shift(1)

    # Drop NaNs created by shifting (edges of each subject's recording)
    dataset = dataset.dropna()

    print(f"\nTotal epochs after filtering: {len(dataset)}")
    print(f"\nSleep stage distribution:")
    print(dataset['sleep_stage'].value_counts().sort_index())

    return dataset, subject_ids


def create_xgboost_objective(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    use_group_kfold: bool = True,
    use_class_weights: bool = False  # DISABLED by default - helps accuracy without motion data
) -> callable:
    """
    Create an Optuna objective function for XGBoost hyperparameter tuning.

    Uses GroupKFold to prevent subject leakage (epochs from same subject
    don't appear in both train and test).
    """

    def objective(trial: optuna.Trial) -> float:
        # Define hyperparameter search space
        params = {
            'objective': 'multi:softmax',
            'num_class': len(np.unique(y)),
            'eval_metric': 'mlogloss',
            'booster': 'gbtree',  # Fixed to gbtree (dart is 10-50x slower)
            'device': 'cuda',     # Use GPU if available (falls back to CPU)
            'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'random_state': 42,
            'verbosity': 0
        }

        # Optionally use class weights (disabled by default for HR-only data)
        sample_weights = None
        if use_class_weights:
            class_weights = {}
            for cls in np.unique(y):
                class_weights[cls] = len(y) / (len(np.unique(y)) * np.sum(y == cls))
            sample_weights = np.array([class_weights[yi] for yi in y])

        # Cross-validation
        if use_group_kfold:
            cv = GroupKFold(n_splits=n_folds)
            splits = cv.split(X, y, groups)
        else:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = cv.split(X, y)

        f1_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            weights_train = sample_weights[train_idx] if sample_weights is not None else None

            # Create and train model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                sample_weight=weights_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Predict and evaluate
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='macro')
            f1_scores.append(f1)

            # Report intermediate value for pruning
            trial.report(np.mean(f1_scores), fold_idx)

            # Prune if not promising
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(f1_scores)

    return objective


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    best_params: Dict,
    sample_weights: Optional[np.ndarray] = None
) -> xgb.XGBClassifier:
    """Train the final model with best hyperparameters."""

    # Remove non-XGBoost parameters
    model_params = {k: v for k, v in best_params.items()}
    model_params['objective'] = 'multi:softmax'
    model_params['num_class'] = len(np.unique(y))
    model_params['random_state'] = 42
    model_params['verbosity'] = 0

    model = xgb.XGBClassifier(**model_params)
    model.fit(X, y, sample_weight=sample_weights)

    return model


def run_optimization(
    dataset: pd.DataFrame,
    n_trials: int = 100,
    n_folds: int = 5,
    use_group_kfold: bool = True
) -> Tuple[optuna.Study, xgb.XGBClassifier]:
    """
    Run Optuna optimization and train final model.
    """
    # Prepare features and labels
    feature_cols = [c for c in dataset.columns if c.startswith('hr_')]
    X = dataset[feature_cols].values

    # Encode sleep stages to consecutive integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(dataset['sleep_stage'].values)

    # Get subject groups for GroupKFold
    subject_encoder = LabelEncoder()
    groups = subject_encoder.fit_transform(dataset['subject_id'].values)

    print(f"\nFeatures shape: {X.shape}")
    print(f"Classes: {label_encoder.classes_} -> {np.unique(y)}")
    print(f"Number of subjects: {len(np.unique(groups))}")

    # Fill NaN values with column medians
    for i in range(X.shape[1]):
        col_median = np.nanmedian(X[:, i])
        X[np.isnan(X[:, i]), i] = col_median

    # Create Optuna study
    print(f"\n{'='*60}")
    print("Starting Optuna Hyperparameter Optimization")
    print(f"{'='*60}")
    print(f"Number of trials: {n_trials}")
    print(f"Cross-validation folds: {n_folds}")
    print(f"Using GroupKFold (prevent subject leakage): {use_group_kfold}")

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    )

    objective = create_xgboost_objective(X, y, groups, n_folds, use_group_kfold)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1  # Use 1 job to avoid memory issues; increase if you have resources
    )

    print(f"\n{'='*60}")
    print("Optimization Complete!")
    print(f"{'='*60}")
    print(f"Best trial:")
    print(f"  Value (Macro F1): {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    # Train final model with best parameters
    print(f"\nTraining final model with best parameters...")

    # Calculate sample weights
    class_weights = {}
    for cls in np.unique(y):
        class_weights[cls] = len(y) / (len(np.unique(y)) * np.sum(y == cls))
    sample_weights = np.array([class_weights[yi] for yi in y])

    final_model = train_final_model(X, y, study.best_params, sample_weights)

    # Final evaluation with leave-one-subject-out
    print(f"\nFinal Evaluation (GroupKFold):")
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))

    all_y_true = []
    all_y_pred = []

    for train_idx, val_idx in cv.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        weights_train = sample_weights[train_idx]

        model_params = study.best_params.copy()
        model_params['objective'] = 'multi:softmax'
        model_params['num_class'] = len(np.unique(y))
        model_params['random_state'] = 42
        model_params['verbosity'] = 0

        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train, sample_weight=weights_train)

        y_pred = model.predict(X_val)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

    # Print classification report
    print("\nClassification Report:")
    stage_names = [SLEEP_STAGE_NAMES[s] for s in label_encoder.classes_]
    print(classification_report(all_y_true, all_y_pred, target_names=stage_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"       {' '.join([f'{s:>6}' for s in stage_names])}")
    for i, row in enumerate(cm):
        print(f"{stage_names[i]:>6} {' '.join([f'{v:>6}' for v in row])}")

    # Feature importance
    print("\nFeature Importance (Top 10):")
    importance = final_model.feature_importances_
    feature_importance = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    for feat, imp in feature_importance[:10]:
        print(f"  {feat}: {imp:.4f}")

    return study, final_model, label_encoder, feature_cols


def main():
    """Main execution function."""
    print("="*60)
    print("Sleep Stage Prediction with Optuna + XGBoost")
    print("="*60)
    print("\nUsing only heart rate data to predict sleep stages")
    print("Stages: Wake (0), N1 (1), N2 (2), N3 (3), REM (5)")

    # Load and prepare dataset
    print("\n" + "="*60)
    print("Loading and preparing dataset...")
    print("="*60)

    dataset, subject_ids = prepare_dataset()

    # Run optimization
    study, model, label_encoder, feature_cols = run_optimization(
        dataset,
        n_trials=100,  # Adjust based on available time/resources
        n_folds=5,
        use_group_kfold=True  # Prevent subject leakage
    )

    # Save results
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)

    # Save best parameters
    best_params_df = pd.DataFrame([study.best_params])
    best_params_df.to_csv(DATA_DIR / "best_params.csv", index=False)
    print(f"Best parameters saved to: best_params.csv")

    # Save model
    model.save_model(str(DATA_DIR / "sleep_stage_model.json"))
    print(f"Model saved to: sleep_stage_model.json")

    # Save optimization history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(DATA_DIR / "optimization_history.csv", index=False)
    print(f"Optimization history saved to: optimization_history.csv")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

    return study, model


if __name__ == "__main__":
    main()
