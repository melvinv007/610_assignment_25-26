import pandas as pd
import numpy as np
import scipy.stats
import os
import gc

base_path = r'C:\Users\vmelv\Downloads\610_assignment\oulad_dataset'
out_data_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\data'
out_res_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\results'
os.makedirs(out_data_path, exist_ok=True)
os.makedirs(out_res_path, exist_ok=True)

print("1. Loading studentInfo.csv")
df_info = pd.read_csv(os.path.join(base_path, 'studentInfo.csv'))
print("Initial shape:", df_info.shape)

# Impute imd_band
mode_imd = df_info['imd_band'].mode()[0]
df_info['imd_band'] = df_info['imd_band'].fillna(mode_imd)

def map_imd(x):
    if pd.isna(x): return x
    x = str(x)
    if '0-10' in x: return 0
    if '10-20' in x: return 1
    if '20-30' in x: return 2
    if '30-40' in x: return 3
    if '40-50' in x: return 4
    if '50-60' in x: return 5
    if '60-70' in x: return 6
    if '70-80' in x: return 7
    if '80-90' in x: return 8
    if '90-100' in x: return 9
    return -1

df_info['imd_band'] = df_info['imd_band'].apply(map_imd)

edu_map = {'No Formal Quals':0, 'Lower Than A Level':1, 'A Level or Equivalent':2, 'HE Qualification':3, 'Post Graduate Qualification':4}
df_info['highest_education'] = df_info['highest_education'].map(edu_map)

age_map = {'0-35':0, '35-55':1, '55<=':2}
df_info['age_band'] = df_info['age_band'].map(age_map)

df_info['gender'] = df_info['gender'].map({'M':1, 'F':0})
df_info['disability'] = df_info['disability'].map({'Y':1, 'N':0})

df_info['binary_result'] = df_info['final_result'].map(lambda x: 1 if x in ['Pass', 'Distinction'] else 0)

print("2. Loading studentRegistration.csv")
df_reg = pd.read_csv(os.path.join(base_path, 'studentRegistration.csv'))
df_reg['is_withdrawn'] = (~df_reg['date_unregistration'].isna()).astype(int)
df_reg['early_registrant'] = (df_reg['date_registration'] < 0).astype(int)
df_info = df_info.merge(df_reg[['id_student', 'code_module', 'code_presentation', 'date_registration', 'is_withdrawn', 'early_registrant']], 
                        on=['id_student', 'code_module', 'code_presentation'], how='left')

print("3. Processing vle.csv and studentVle.csv")
df_vle = pd.read_csv(os.path.join(base_path, 'vle.csv'))

chunksize = 500000
weekly_clicks = []
activity_clicks = []

for i, chunk in enumerate(pd.read_csv(os.path.join(base_path, 'studentVle.csv'), chunksize=chunksize)):
    print(f"  Processing chunk {i+1}...")
    chunk = chunk.merge(df_vle[['id_site', 'activity_type']], on='id_site', how='left')
    chunk['week_number'] = chunk['date'] // 7
    chunk['week_number'] = chunk['week_number'].clip(-4, 40)
    
    # Aggregate by week
    agg_week = chunk.groupby(['id_student', 'code_module', 'code_presentation', 'week_number'], as_index=False)['sum_click'].sum()
    weekly_clicks.append(agg_week)
    
    # Aggregate by activity type
    agg_act = chunk.groupby(['id_student', 'code_module', 'code_presentation', 'activity_type'], as_index=False)['sum_click'].sum()
    activity_clicks.append(agg_act)

df_weekly = pd.concat(weekly_clicks).groupby(['id_student', 'code_module', 'code_presentation', 'week_number'], as_index=False)['sum_click'].sum()
df_activity = pd.concat(activity_clicks).groupby(['id_student', 'code_module', 'code_presentation', 'activity_type'], as_index=False)['sum_click'].sum()

print("4. Creating weekly click matrix")
df_weekly_pivot = df_weekly.pivot_table(index=['id_student', 'code_module', 'code_presentation'], columns='week_number', values='sum_click', fill_value=0)
df_weekly_pivot.columns = [f'Wpre{abs(c)}' if c < 0 else f'W{c}' for c in df_weekly_pivot.columns]
df_weekly_pivot = df_weekly_pivot.reset_index()

# Add temporal features
pos_weeks = [c for c in df_weekly_pivot.columns if c.startswith('W') and not c.startswith('Wpre')]
w1_to_w4 = [c for c in pos_weeks if int(c[1:]) <= 4]
w17_to_w30 = [c for c in pos_weeks if 17 <= int(c[1:]) <= 30]

df_weekly_pivot['total_clicks'] = df_weekly_pivot[[c for c in df_weekly_pivot.columns if c.startswith('W')]].sum(axis=1)
df_weekly_pivot['active_weeks'] = (df_weekly_pivot[[c for c in df_weekly_pivot.columns if c.startswith('W')]] > 0).sum(axis=1)
df_weekly_pivot['early_clicks'] = df_weekly_pivot[w1_to_w4].sum(axis=1) if w1_to_w4 else 0
df_weekly_pivot['late_clicks'] = df_weekly_pivot[w17_to_w30].sum(axis=1) if w17_to_w30 else 0
df_weekly_pivot['early_ratio'] = df_weekly_pivot['early_clicks'] / (df_weekly_pivot['total_clicks'] + 1)
df_weekly_pivot['regularity'] = df_weekly_pivot[pos_weeks].std(axis=1)
df_weekly_pivot['peak_week'] = df_weekly_pivot[pos_weeks].idxmax(axis=1).apply(lambda x: int(x[1:]) if isinstance(x, str) else -1)

df_weekly_pivot.to_csv(os.path.join(out_data_path, 'oulad_weekly_clicks.csv'), index=False)

print("5. Processing assessment latency")
df_assessments = pd.read_csv(os.path.join(base_path, 'assessments.csv'))
df_assessments = df_assessments[df_assessments['assessment_type'] == 'TMA'].copy()
df_student_assessment = pd.read_csv(os.path.join(base_path, 'studentAssessment.csv'))
df_latency = df_student_assessment.merge(df_assessments, on='id_assessment', how='inner')

df_latency['due_date'] = df_latency['date'].astype(float)
df_latency['open_date'] = np.maximum(0, df_latency['due_date'] - 30)
df_latency['date_submitted'] = df_latency['date_submitted'].astype(float)
df_latency['relative_latency'] = (df_latency['date_submitted'] - df_latency['open_date']) / (df_latency['due_date'] - df_latency['open_date'])
df_latency['relative_latency'] = df_latency['relative_latency'].clip(0, 1.5)
df_latency['missing_submission'] = df_latency['date_submitted'].isna()
df_latency.loc[df_latency['missing_submission'], 'relative_latency'] = 1.0

df_latency = df_latency.sort_values(['id_student', 'code_module', 'code_presentation', 'due_date'])

latency_features = []
for (sid, mod, pres), group in df_latency.groupby(['id_student', 'code_module', 'code_presentation']):
    tma_count = len(group)
    missing = group['missing_submission'].sum()
    mean_lat = group['relative_latency'].mean()
    if tma_count >= 3:
        y = group['relative_latency'].values
        x = np.arange(len(y))
        slope, _, _, _, _ = scipy.stats.linregress(x, y)
        tau, _ = scipy.stats.kendalltau(x, y)
    else:
        slope, tau = np.nan, np.nan
    
    if slope > 0.05: trend = 'Worsening'
    elif slope < -0.05: trend = 'Improving'
    else: trend = 'Stable'
    
    latency_features.append({
        'id_student': sid, 'code_module': mod, 'code_presentation': pres,
        'latency_slope': slope, 'latency_tau': tau, 'latency_trend': trend,
        'tma_count': tma_count, 'missing_submissions': missing, 'mean_latency': mean_lat
    })

df_lat_feat = pd.DataFrame(latency_features)
df_lat_feat.to_csv(os.path.join(out_data_path, 'oulad_latency.csv'), index=False)

print("6. Building Master DataFrame")
df_activity_pivot = df_activity.pivot_table(index=['id_student', 'code_module', 'code_presentation'], columns='activity_type', values='sum_click', fill_value=0).reset_index()

df_master = df_info.merge(df_weekly_pivot, on=['id_student', 'code_module', 'code_presentation'], how='left')
df_master = df_master.merge(df_lat_feat, on=['id_student', 'code_module', 'code_presentation'], how='left')
df_master = df_master.merge(df_activity_pivot, on=['id_student', 'code_module', 'code_presentation'], how='left')

# Drop those without any clicks as they didn't really participate
df_master = df_master[df_master['total_clicks'].notna()].copy()

# Encode code_module now
df_master = pd.get_dummies(df_master, columns=['code_module'], drop_first=True)

df_master.to_csv(os.path.join(out_data_path, 'oulad_master.csv'), index=False)

print("7. Generating Report")
with open(os.path.join(out_res_path, 'preprocessing_report.txt'), 'w') as f:
    f.write("OULAD Preprocessing Report\n==========================\n")
    f.write(f"Final Master DataFrame Shape: {df_master.shape}\n\n")
    f.write("Class Distribution (final_result):\n")
    f.write(df_master['final_result'].value_counts().to_string() + "\n\n")
    f.write("Binary Result (1=Pass/Dist, 0=Fail/Withdrawn):\n")
    f.write(df_master['binary_result'].value_counts().to_string() + "\n\n")
    f.write(f"Students with Latency Data (>= 3 TMAs): {df_master['latency_slope'].notna().sum()}\n")

print("Done! OULAD preprocessing complete.")
