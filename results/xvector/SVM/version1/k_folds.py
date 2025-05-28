import pandas as pd
from sklearn.model_selection import KFold
from collections import defaultdict
import os

# Load the dataset
train_file = r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\data\xvector_pretrained_dataset_with_embeddings.csv'
df = pd.read_csv(train_file, low_memory=False)

# Use speaker as the grouping element
speakers = df["speaker"].unique()


# Use GroupKFold-like logic
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(speakers)):
    train_speakers = speakers[train_idx]
    test_speakers = speakers[test_idx]

    train_df = df[df["speaker"].isin(train_speakers)]
    test_df = df[df["speaker"].isin(test_speakers)]

    # Save to CSV
    train_df.to_csv(fr"C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\results\train_fold_{fold+1}.csv", index=False)
    test_df.to_csv(fr"C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\results\test_fold_{fold+1}.csv", index=False)
    
    print(f"Fold {fold+1}: Saved train_fold_{fold+1}.csv and test_fold_{fold+1}.csv")
