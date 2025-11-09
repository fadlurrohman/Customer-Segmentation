import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

REFERENCE_DATE = pd.Timestamp('2025-04-01')

def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    df["checkin_date"] = pd.to_datetime(df["checkin_date"])
    agg = df.groupby("customer_id").agg(
        last_booking=("checkin_date", "max"),
        frequency=("checkin_date", "count"),
        monetary=("revenue_usd", "sum")
    )
    agg["recency"] = (REFERENCE_DATE - agg["last_booking"]).dt.days
    rfm = agg[["recency", "frequency", "monetary"]].copy()
    return rfm

def run_clustering(rfm: pd.DataFrame, n_clusters: int = 4):
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(X)
    rfm_clustered = rfm.copy()
    rfm_clustered["cluster"] = labels
    return rfm_clustered, model

def label_clusters(rfm_clustered: pd.DataFrame) -> pd.DataFrame:
    cluster_summary = rfm_clustered.groupby("cluster").agg(
        recency_mean=("recency", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean"),
        count=("recency", "count")
    )

    labels = {}
    for cluster_id, row in cluster_summary.iterrows():
        if row["frequency_mean"] > rfm_clustered["frequency"].median() and row["monetary_mean"] > rfm_clustered["monetary"].median():
            labels[cluster_id] = "VIP / High-Value"
        elif row["recency_mean"] < rfm_clustered["recency"].median():
            labels[cluster_id] = "Active Regulars"
        elif row["monetary_mean"] < rfm_clustered["monetary"].median():
            labels[cluster_id] = "Low-Value / Deal Hunters"
        else:
            labels[cluster_id] = "At-Risk / Dormant"

    rfm_clustered["segment"] = rfm_clustered["cluster"].map(labels)
    return rfm_clustered, cluster_summary

if __name__ == "__main__":
    df = pd.read_csv("booking_data.csv")
    rfm = build_rfm(df)
    rfm_clustered, model = run_clustering(rfm, n_clusters=4)
    rfm_labeled, summary = label_clusters(rfm_clustered)

    print("===== Cluster summary =====")
    print(summary)

    print("\n===== Example of labeled customers =====")
    print(rfm_labeled.head())

    rfm_labeled.to_csv("customer_segments.csv", index=True)
    print("\nSaved customer_segments.csv with RFM scores and segment labels.")
