import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load CSV
df = pd.read_csv("cfb_2005_2024_preprocessed.csv")

# Create qid = seasonYear + weekNumber
df["qid"] = df["seasonYear"].astype(str) + df["weekNumber"].astype(str).str.zfill(2)

# Save true labels separately for testing (but not used in training)
df["true_ap_label"] = df["apRank"].apply(lambda x: 25 - x if pd.notnull(x) and x <= 25 else 0)

# For training, we'll use dummy label (0) – model will learn to rank from features
df["label"] = 0

# Drop non-numeric / non-feature columns
drop_cols = [
    "opponent_teamName", "apRank", "coachesRank",
    "apRankTrend", "coachesRankTrend", "nextWeekRank", "wasRankedLastWeek",
    "opponent_apRank", "opponent_coachesRank", "opponent_apRankTrend",
    "opponent_coachesRankTrend", "opponent_nextWeekRank", "opponent_wasRankedLastWeek"
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Split metadata for evaluation
qid_ap_labels = df[["qid", "true_ap_label"]].copy()

# Normalize features
features = df.drop(columns=["label", "true_ap_label", "qid"])
scaler = MinMaxScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# Rebuild with scaled features
df_final = pd.concat([df[["label", "qid"]].reset_index(drop=True), features_scaled], axis=1)

# Write to LibSVM
def to_libsvm(df, filename):
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            label = int(row["label"])  # always 0
            qid = row["qid"]
            feats = " ".join([f"{i + 1}:{v:.6f}" for i, v in enumerate(row[2:])])
            f.write(f"{label} qid:{qid} {feats}\n")

to_libsvm(df_final, "predict_ap_ranking.libsvm")
qid_ap_labels.to_csv("true_ap_labels.csv", index=False)
