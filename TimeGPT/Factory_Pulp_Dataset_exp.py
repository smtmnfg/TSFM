# Install necessary package
!pip install -Uqq nixtla

# Import necessary libraries
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error
from datetime import datetime
import pandas as pd
import numpy as np
from nixtla import NixtlaClient

# Function to compute classification metrics
def compute_metrics(df, predicted_variable, actual_variable):
    y_true = df[actual_variable]
    y_pred = df[predicted_variable]

    metrics = {
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
        'accuracy': accuracy_score(y_true, y_pred)
    }

    return metrics

# Function to evaluate forecasting metrics
def evaluate_forecasts(df):
    y_true = df['y']
    y_pred = df['TimeGPT']

    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }

    return metrics

# Function to perform train-test split for forecasting
def train_test_split(df):
    train = pd.DataFrame()
    test = pd.DataFrame()
    unique_ids = df['unique_id'].unique().tolist()
    split_threshold = int(0.8 * df[df['unique_id'] == unique_ids[0]].shape[0])

    for id in unique_ids:
        id_df = df[df['unique_id'] == id].reset_index()
        train = pd.concat([train, id_df.iloc[:split_threshold]])
        test = pd.concat([test, id_df.iloc[split_threshold:]])

    return train.reset_index(drop=True), test.reset_index(drop=True)

# Initialize Nixtla client
nixtla_client = NixtlaClient(api_key='your-nixtla-api-key')


"""TimeGPT on Pulp Dataset"""

# Load and preprocess Pulp dataset
# Dropping categorical variables from the dataset
df_pulp = pd.read_csv("./PulpDataSet/processminer-rare-event-mts - data.csv").drop(['x28', 'x61'], axis=1)
columns = [col for col in df_pulp.columns if col not in ['time', 'y']]

# Melt dataset to long format
df_pulp_pivot = pd.melt(df_pulp, id_vars=['y', 'time'], value_vars=columns, var_name='unique_id', value_name='values')
df_pulp_pivot['ds'] = pd.to_datetime(df_pulp_pivot['time'])

# ------------------------------------ Anomaly Detection on Pulp Dataset ------------------------------------
anomalies_df = nixtla_client.detect_anomalies(
    df=df_pulp_pivot[['unique_id', 'values', 'y', 'ds']],
    time_col='ds',
    target_col='y',
    freq='2min'
)

anomalies_df['anomaly'] = anomalies_df['anomaly'].astype(int)
classification_metrics_anomaly_detection = compute_metrics(anomalies_df, 'anomaly', 'y')
print("Pulp Dataset Anomaly Detection Metrics:", classification_metrics_anomaly_detection)
# ------------------------------------------------------------------------------------------------------------

# Train-test split and forecast on Pulp dataset
pulp_train, pulp_test = train_test_split(df_pulp_pivot)

# ------------------------------------ Forecasting on Pulp Dataset ------------------------------------
fcst_df = nixtla_client.forecast(
    df=pulp_train,
    h=pulp_test[pulp_test['unique_id'] == 'x1'].shape[0],
    level=[90],
    finetune_steps=20,
    finetune_loss='mae',
    model='timegpt-1-long-horizon',
    time_col='ds',
    target_col='values',
    id_col='unique_id',
    freq='2min'
)

pulp_test = pd.merge(pulp_test, fcst_df[['unique_id', 'ds', 'TimeGPT']], on=['unique_id', 'ds'])
forecast_metrics = evaluate_forecasts(pulp_test.dropna())
print("Pulp Dataset Forecasting Metrics:", forecast_metrics)
# ------------------------------------------------------------------------------------------------------------


"""TimeGPT on Factory Dataset"""

# Load and preprocess Factory dataset
df_ff = pd.read_csv("./FFDataset/FF_Analog.csv")
df_ff['actual_state'] = df_ff['actual_state'].map({'normal': 0, 'E_STOPPED': 1, 'NoNose': 1, 'NoBody2': 1, 'NoNose,NoBody2': 1, 'NoBody1': 1, 'NoBody2,NoBody1': 1, 'NoNose,NoBody2,NoBody1': 1})
df_ff['timestamp'] = pd.to_datetime(df_ff['_time'], format='ISO8601')

df_ff_important_features = df_ff[['timestamp', 'I_R02_Gripper_Pot', 'I_R03_Gripper_Pot', 'I_R03_Gripper_Load', 'actual_state']].resample('1T', on='timestamp').agg({
    'I_R02_Gripper_Pot': 'mean',
    'I_R03_Gripper_Pot': 'mean',
    'I_R03_Gripper_Load': 'mean',
    'actual_state': 'max'
}).interpolate().reset_index()

df_ff_important_features_pivot = pd.melt(df_ff_important_features,
    id_vars=['timestamp', 'actual_state'],
    value_vars=['I_R02_Gripper_Pot', 'I_R03_Gripper_Pot', 'I_R03_Gripper_Load'],
    var_name='unique_id',
    value_name='values')

# ------------------------------------ Anomaly Detection on Factory Dataset ------------------------------------
anomalies_df = nixtla_client.detect_anomalies(
    df=df_ff_important_features_pivot[['unique_id', 'values', 'actual_state', 'timestamp']],
    time_col='timestamp',
    target_col='actual_state',
    freq='1T'
)

anomalies_df['anomaly'] = anomalies_df['anomaly'].astype(int)
classification_metrics_anomaly_detection = compute_metrics(anomalies_df, 'anomaly', 'actual_state')
print("Factory Dataset Anomaly Detection Metrics:", classification_metrics_anomaly_detection)
# ------------------------------------------------------------------------------------------------------------

# Train-test split and forecast on Factory dataset
FF_train, FF_test = train_test_split(df_ff_important_features_pivot)

# ------------------------------------ Forecasting on Factory Dataset ------------------------------------
fcst_df = nixtla_client.forecast(
    df=FF_train,
    h=FF_test[FF_test['unique_id'] == 'I_R02_Gripper_Pot'].shape[0],
    level=[90],
    finetune_steps=20,
    finetune_loss='mae',
    model='timegpt-1-long-horizon',
    time_col='timestamp',
    target_col='values',
    id_col='unique_id'
)

FF_test = pd.merge(FF_test, fcst_df[['unique_id', 'timestamp', 'TimeGPT']], on=['unique_id', 'timestamp'])
forecast_metrics = evaluate_forecasts(FF_test.dropna())
print("Factory Dataset Forecasting Metrics:", forecast_metrics)
# ------------------------------------------------------------------------------------------------------------
