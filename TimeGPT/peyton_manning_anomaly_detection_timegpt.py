#| hide
!pip install -Uqq nixtla

#| hide
from nixtla.utils import in_colab

#| hide
IN_COLAB = in_colab()

#| hide
if not IN_COLAB:
    from nixtla.utils import colab_badge
    from dotenv import load_dotenv

"""# Anomaly detection


"""

import pandas as pd
from nixtla import NixtlaClient
from sklearn.metrics import classification_report

nixtla_client = NixtlaClient(
    # defaults to os.environ.get("NIXTLA_API_KEY")
    api_key = 'ADD YOUR API KEY HERE'
)

"""## Load dataset


"""

df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/peyton_manning.csv')

df.head()

nixtla_client.plot(
    df,
    time_col='timestamp',
    target_col='value',
    max_insample_length=365
)

"""## Anomaly detection"""

anomalies_df = nixtla_client.detect_anomalies(
    df,
    time_col='timestamp',
    target_col='value',
    freq='D'
)

anomalies_df

nixtla_client.plot(
    df,
    anomalies_df,
    time_col='timestamp',
    target_col='value'
)

"""## Modifying the confidence intervals

"""

anomalies_df = nixtla_client.detect_anomalies(
    df,
    time_col='timestamp',
    target_col='value',
    freq='D',
    level=90
)

anomalies_df.anomaly.value_counts()

anomalies_df2 = nixtla_client.detect_anomalies(
    df,
    time_col='timestamp',
    target_col='value',
    freq='D',
    level=60
)

anomalies_df2


# Ensure that both anomaly columns are boolean or binary (True/False or 1/0)
anomalies_df2['anomaly'] = anomalies_df2['anomaly'].astype(int)
anomalies_df['anomaly'] = anomalies_df['anomaly'].astype(int)

# Create the classification report
report = classification_report(anomalies_df['anomaly'], anomalies_df2['anomaly'], target_names=['False', 'True'])

# Print the classification report
print(report)

nixtla_client.plot(
    anomalies_df,
    anomalies_df2,
    time_col='timestamp',
    target_col='value'
)

