import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from config import LOW_VARIANCE_THRESHOLD

# Add hour/day features, drop raw Time
def convert_time_features(df):
    df = df.copy()
    df["Hours"] = (df["Time"] // 3600) % 24
    df["Days"] = (df["Time"] // (3600 * 24)) % 7
    df.drop(columns=["Time"], inplace=True)
    return df

# Scale and remove low variance
def scale_and_select_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = VarianceThreshold(threshold=LOW_VARIANCE_THRESHOLD)
    X_selected = selector.fit_transform(X_scaled)

    return X_selected , scaler
