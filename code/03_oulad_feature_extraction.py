import pandas as pd
import numpy as np
import scipy.stats
import os

out_data_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\data'

print("1. Loading Data")
df_master = pd.read_csv(os.path.join(out_data_path, 'oulad_master.csv'))
df_weekly = pd.read_csv(os.path.join(out_data_path, 'oulad_weekly_clicks.csv'))

print("2. Normalizing Weekly Vectors for DTW")
pos_weeks = [c for c in df_weekly.columns if c.startswith('W') and not c.startswith('Wpre')]
norm_vectors = []

# To normalize course length, we map weeks 1..N to 20 evenly spaced points
# For simplicity and given OULAD lengths (around 38 weeks), we'll interpolate based on actual active weeks
for idx, row in df_weekly.iterrows():
    vec = row[pos_weeks].values.astype(float)
    max_click = np.max(vec)
    if max_click > 0:
        vec_norm = vec / max_click
    else:
        vec_norm = vec
    
    # Interpolate to exactly 20 points
    old_x = np.linspace(0, 1, len(vec_norm))
    new_x = np.linspace(0, 1, 20)
    vec_interp = np.interp(new_x, old_x, vec_norm)
    
    norm_vectors.append([row['id_student'], row['code_module'], row['code_presentation']] + list(vec_interp))

df_norm = pd.DataFrame(norm_vectors, columns=['id_student', 'code_module', 'code_presentation'] + [f'T{i+1}' for i in range(20)])
df_norm.to_csv(os.path.join(out_data_path, 'oulad_weekly_normalized.csv'), index=False)

print("3. Final Feature Matrix")
df_features = df_master.copy()

trend_map = {'Worsening': 1, 'Stable': 0, 'Improving': -1}
df_features['latency_trend'] = df_features['latency_trend'].map(trend_map).fillna(0)

cols_to_keep = ['id_student', 'code_presentation', 'binary_result', 'final_result',
                'total_clicks', 'active_weeks', 'early_clicks', 'late_clicks', 'early_ratio', 'regularity', 'peak_week',
                'latency_slope', 'latency_tau', 'latency_trend', 'missing_submissions', 'mean_latency',
                'gender', 'age_band', 'imd_band', 'highest_education', 'disability']

# add module dummies
mod_cols = [c for c in df_features.columns if c.startswith('code_module_') and c != 'code_module_presentation']
cols_to_keep.extend(mod_cols)
# add activity columns
act_cols = ['forumng', 'oucontent', 'quiz', 'subpage', 'homepage', 'resource', 'url']
act_cols = [c for c in act_cols if c in df_features.columns]
cols_to_keep.extend(act_cols)

df_features = df_features[cols_to_keep].copy()
df_features = df_features.dropna(subset=['binary_result', 'total_clicks'])

df_features.to_csv(os.path.join(out_data_path, 'oulad_features_ml.csv'), index=False)
print(f"Final ML features shape: {df_features.shape}")
print("Done! OULAD feature extraction complete.")
