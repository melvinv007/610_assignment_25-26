import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

out_data_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\data'
fig_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\figures'
res_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\results'

print("1. Loading ML features")
df = pd.read_csv(os.path.join(out_data_path, 'moodle_features_ml.csv'))

# Prep for classification
y = df['submitted_any_assignment']
X = df.drop(columns=['User ID', 'submitted_any_assignment', 'high_engagement', 'TDI_quartile'], errors='ignore')
X = X.fillna(0)

# Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("2. Logistic Regression (H3, H4 Validation)")
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_scaled, y)
lr_preds = lr.predict(X_scaled)
lr_probs = lr.predict_proba(X_scaled)[:, 1]

with open(os.path.join(res_path, 'moodle_ml_results.txt'), 'w') as f:
    f.write("=== Moodle Logistic Regression (Predicting Assignment Submission) ===\n")
    f.write(f"Accuracy: {accuracy_score(y, lr_preds):.4f}\n")
    f.write(f"F1 Macro: {f1_score(y, lr_preds, average='macro'):.4f}\n")
    f.write(f"ROC AUC: {roc_auc_score(y, lr_probs):.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y, lr_preds) + "\n\n")
    f.write("Coefficients:\n")
    for col, coef in zip(X.columns, lr.coef_[0]):
        f.write(f"{col}: {coef:.4f}\n")

print("3. K-Means Clustering")
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)
sil = silhouette_score(X_scaled, labels)

df['cluster'] = labels

with open(os.path.join(res_path, 'moodle_ml_results.txt'), 'a') as f:
    f.write(f"\n=== Moodle K-Means (k=3) ===\n")
    f.write(f"Silhouette Score: {sil:.4f}\n")
    for i in range(3):
        f.write(f"Cluster {i} size: {sum(labels==i)} (Submit %: {df[df['cluster']==i]['submitted_any_assignment'].mean():.2f})\n")

print("4. K-Means Profile Plot")
cluster_means = df.groupby('cluster')[['total_events', 'TDI', 'total_sessions', 'content_diversity_entropy']].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(scaler.fit_transform(cluster_means), annot=True, cmap='coolwarm', yticklabels=[f'C{i}' for i in range(3)], xticklabels=cluster_means.columns)
plt.title("Moodle Cluster Centroids (Standardized)")
plt.tight_layout()
plt.savefig(os.path.join(fig_path, 'fig13_moodle_cluster_profiles.png'), bbox_inches='tight')
plt.close()

print("Done! Moodle ML complete.")
