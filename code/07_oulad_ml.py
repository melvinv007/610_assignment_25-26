import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

out_data_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\data'
fig_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\figures'
res_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\results'

print("1. Loading ML features")
df = pd.read_csv(os.path.join(out_data_path, 'oulad_features_ml.csv'))

# Target and predictors
y = df['binary_result']
X = df.drop(columns=['id_student', 'code_module', 'code_presentation', 'binary_result', 'final_result', 'date_registration', 'is_withdrawn'], errors='ignore')
X = X.fillna(0)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("2. Logistic Regression (H1)")
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]

with open(os.path.join(res_path, 'oulad_ml_results.txt'), 'w') as f:
    f.write("=== OULAD Logistic Regression ===\n")
    f.write(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}\n")
    f.write(f"F1 Macro: {f1_score(y_test, lr_preds, average='macro'):.4f}\n")
    f.write(f"ROC AUC: {roc_auc_score(y_test, lr_probs):.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, lr_preds) + "\n\n")

print("3. Random Forest")
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

with open(os.path.join(res_path, 'oulad_ml_results.txt'), 'a') as f:
    f.write("=== OULAD Random Forest ===\n")
    f.write(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}\n")
    f.write(f"F1 Macro: {f1_score(y_test, rf_preds, average='macro'):.4f}\n")
    f.write(f"ROC AUC: {roc_auc_score(y_test, rf_probs):.4f}\n\n")

print("4. Feature Importance Plot (RF)")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(10, 6))
plt.title("Top 20 Feature Importances (Random Forest)")
plt.bar(range(20), importances[indices], align="center")
plt.xticks(range(20), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(fig_path, 'fig11_oulad_rf_feature_importance.png'), bbox_inches='tight')
plt.close()

print("5. K-Means Trajectory Clustering (H2)")
try:
    df_norm = pd.read_csv(os.path.join(out_data_path, 'oulad_weekly_normalized.csv'))
    T_cols = [c for c in df_norm.columns if c.startswith('T')]
    X_clust = df_norm[T_cols].fillna(0).values
    
    # Subsample for faster kmeans if needed
    if len(X_clust) > 10000:
        np.random.seed(42)
        idx = np.random.choice(len(X_clust), 10000, replace=False)
        X_fit = X_clust[idx]
    else:
        X_fit = X_clust
        
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_fit)
    labels = kmeans.predict(X_clust)
    
    df_norm['cluster'] = labels
    
    plt.figure(figsize=(10, 6))
    for i in range(4):
        cluster_mean = df_norm[df_norm['cluster'] == i][T_cols].mean().values
        plt.plot(range(1, 21), cluster_mean, label=f'Cluster {i} (n={sum(labels==i)})')
    plt.title('OULAD Weekly Trajectory Archetypes (K-Means)')
    plt.xlabel('Normalized Course Time')
    plt.ylabel('Normalized Clicks')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'fig12_oulad_kmeans_trajectories.png'), bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(res_path, 'oulad_ml_results.txt'), 'a') as f:
        f.write("=== OULAD K-Means Trajectory Clustering ===\n")
        f.write(f"Cluster counts: {np.bincount(labels)}\n\n")
except Exception as e:
    print(f"K-means skipped: {e}")

print("Done! OULAD ML complete.")
