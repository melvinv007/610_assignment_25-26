import pandas as pd
import numpy as np
import scipy.stats
import os

out_data_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\data'
out_res_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\results'

print("1. Loading Data")
df_feats = pd.read_csv(os.path.join(out_data_path, 'moodle_student_features.csv'))
df_weekly = pd.read_csv(os.path.join(out_data_path, 'moodle_weekly.csv'))

print("2. H3 - TDI Analysis")
df_tdi = df_feats[df_feats['total_quiz_attempts'] >= 5].copy()
df_tdi['TDI_quartile'] = pd.qcut(df_tdi['TDI'], 4, labels=[1, 2, 3, 4])
df_feats = df_feats.merge(df_tdi[['User ID', 'TDI_quartile']], on='User ID', how='left')

top_q = df_tdi[df_tdi['TDI_quartile'] == 4]['assessment_submission_rate']
bot_q = df_tdi[df_tdi['TDI_quartile'] == 1]['assessment_submission_rate']

if len(top_q) > 0 and len(bot_q) > 0:
    stat, p_val = scipy.stats.mannwhitneyu(top_q, bot_q)
    s_corr, p_corr = scipy.stats.spearmanr(df_tdi['TDI'], df_tdi['assessment_submission_rate'])
    
    with open(os.path.join(out_res_path, 'h3_tdi_stats.txt'), 'w') as f:
        f.write(f"H3 TDI Analysis\n================\n")
        f.write(f"Mann-Whitney U top vs bottom quartile: p={p_val:.4f}\n")
        f.write(f"Spearman corr (TDI vs submission rate): r={s_corr:.4f}, p={p_corr:.4f}\n")

print("3. H4 - Change-Point Detection (Fallback Sliding Window)")
# Fallback method since ruptures might not be installed
changepoints = []
for idx, row in df_weekly.iterrows():
    uid = row['User ID']
    lrn_cols = [c for c in df_weekly.columns if c.startswith('learning_events_W')]
    lrn_cols = sorted(lrn_cols, key=lambda x: int(x.split('_W')[1]))
    
    vals = row[lrn_cols].values
    collapse_week = -1
    has_collapse = 0
    
    if len(vals) >= 6:
        base_mean = np.mean(vals[:3])
        if base_mean > 0:
            for w in range(3, len(vals) - 2):
                roll_mean = np.mean(vals[w:w+3])
                if roll_mean < 0.3 * base_mean:
                    collapse_week = int(lrn_cols[w].split('_W')[1])
                    has_collapse = 1
                    break
    
    decoupler = 0
    if has_collapse:
        # Check quiz events stability
        ass_cols = [c for c in df_weekly.columns if c.startswith('assessment_events_W')]
        ass_cols = sorted(ass_cols, key=lambda x: int(x.split('_W')[1]))
        w_idx = int(collapse_week) - 1 # 1-indexed to 0-indexed approx
        if w_idx < len(ass_cols):
            pre_ass = np.mean(row[ass_cols[:w_idx]])
            post_ass = np.mean(row[ass_cols[w_idx:]])
            if post_ass >= 0.5 * pre_ass:
                decoupler = 1
                
    changepoints.append({
        'User ID': uid, 'has_content_collapse': has_collapse,
        'learning_collapse_week': collapse_week, 'decoupler': decoupler
    })

df_cp = pd.DataFrame(changepoints)
df_cp.to_csv(os.path.join(out_data_path, 'moodle_changepoint_results.csv'), index=False)

print("4. Building Final ML Features")
df_feats = df_feats.merge(df_cp, on='User ID', how='left')

# Add early/late aggregates
lrn_w1_w4 = [c for c in df_weekly.columns if c.startswith('learning_events_W') and int(c.split('_W')[1]) <= 4]
lrn_w8_plus = [c for c in df_weekly.columns if c.startswith('learning_events_W') and int(c.split('_W')[1]) >= 8]
ass_w1_w4 = [c for c in df_weekly.columns if c.startswith('assessment_events_W') and int(c.split('_W')[1]) <= 4]
ass_w8_plus = [c for c in df_weekly.columns if c.startswith('assessment_events_W') and int(c.split('_W')[1]) >= 8]

df_weekly['learning_events_early_weeks'] = df_weekly[lrn_w1_w4].sum(axis=1) if lrn_w1_w4 else 0
df_weekly['learning_events_late_weeks'] = df_weekly[lrn_w8_plus].sum(axis=1) if lrn_w8_plus else 0
df_weekly['assessment_events_early'] = df_weekly[ass_w1_w4].sum(axis=1) if ass_w1_w4 else 0
df_weekly['assessment_events_late'] = df_weekly[ass_w8_plus].sum(axis=1) if ass_w8_plus else 0

df_feats = df_feats.merge(df_weekly[['User ID', 'learning_events_early_weeks', 'learning_events_late_weeks', 'assessment_events_early', 'assessment_events_late']], on='User ID', how='left')

df_feats['tdi_available'] = df_feats['TDI'].notna().astype(int)
df_feats['TDI'] = df_feats['TDI'].fillna(0)

df_feats.to_csv(os.path.join(out_data_path, 'moodle_features_ml.csv'), index=False)

print(f"Final ML features shape: {df_feats.shape}")
print("Done! Moodle feature extraction complete.")
