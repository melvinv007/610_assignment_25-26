import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import os

plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})
COLOR_MAP = {'high_engagement': '#2ecc71', 'low_engagement': '#e74c3c', 'submitted': '#3498db', 'not_submitted': '#95a5a6'}

out_data_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\data'
fig_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\figures'
os.makedirs(fig_path, exist_ok=True)

try:
    df_feats = pd.read_csv(os.path.join(out_data_path, 'moodle_student_features.csv'))
    df_weekly = pd.read_csv(os.path.join(out_data_path, 'moodle_weekly.csv'))
except FileNotFoundError as e:
    print(f"Data files missing: {e}")
    exit(1)

print("Figure 6: Weekly Learning vs Assessment")
try:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    df_merged = df_weekly.merge(df_feats[['User ID', 'submitted_any_assignment']], on='User ID')
    
    for idx, (sub_val, title, ax) in enumerate(zip([1, 0], ['Submitters', 'Non-Submitters'], axes)):
        sub_df = df_merged[df_merged['submitted_any_assignment'] == sub_val]
        lrn_cols = sorted([c for c in sub_df.columns if c.startswith('learning_events_W')], key=lambda x: int(x.split('_W')[1]))
        ass_cols = sorted([c for c in sub_df.columns if c.startswith('assessment_events_W')], key=lambda x: int(x.split('_W')[1]))
        
        lrn_means = sub_df[lrn_cols].mean().values
        ass_means = sub_df[ass_cols].mean().values
        weeks = range(1, len(lrn_means) + 1)
        
        ax.plot(weeks, lrn_means, label='Learning Content', color='blue', linestyle='-')
        ax.plot(weeks, ass_means, label='Assessment', color='orange', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Week Number')
        if idx == 0: ax.set_ylabel('Mean Events per Week')
        ax.legend()
        
    plt.suptitle('Weekly Learning Content vs Assessment Events (H4)')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'fig6_moodle_weekly_learning_vs_assessment.png'), bbox_inches='tight')
    plt.close()
    print("Saved fig6")
except Exception as e:
    print(f"Error fig6: {e}")

print("Figure 7: TDI Distribution")
try:
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3)
    
    ax0 = plt.subplot(gs[0])
    df_feats['submitted_str'] = df_feats['submitted_any_assignment'].map({1: 'submitted', 0: 'not_submitted'})
    sns.histplot(data=df_feats, x='TDI', hue='submitted_str', palette=COLOR_MAP, ax=ax0, multiple='stack')
    ax0.set_title('TDI Histogram')
    
    ax1 = plt.subplot(gs[1])
    if 'TDI_quartile' in df_feats.columns:
        sns.boxplot(x='TDI_quartile', y='assessment_submission_rate', data=df_feats, ax=ax1)
        ax1.set_title('Submission Rate by TDI Quartile')
    else:
        # Fallback if TDI_quartile not in this DF
        df_feats['TDI_quartile_local'] = pd.qcut(df_feats['TDI'], 4, labels=[1, 2, 3, 4])
        sns.boxplot(x='TDI_quartile_local', y='assessment_submission_rate', data=df_feats, ax=ax1)
        ax1.set_title('Submission Rate by TDI Quartile')
        
    ax2 = plt.subplot(gs[2])
    sns.regplot(x='TDI', y='assessment_submission_rate', data=df_feats, scatter=False, ax=ax2, color='k')
    sns.scatterplot(x='TDI', y='assessment_submission_rate', hue='submitted_str', data=df_feats, palette=COLOR_MAP, ax=ax2)
    ax2.set_title('TDI vs Submission Rate')
    
    plt.suptitle('H3: Temporal Distribution Index vs Assessment Engagement')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'fig7_tdi_distribution_and_h3.png'), bbox_inches='tight')
    plt.close()
    print("Saved fig7")
except Exception as e:
    print(f"Error fig7: {e}")

print("Figure 8: Event Heatmap")
try:
    plt.figure(figsize=(18, 10))
    evt_cols = sorted([c for c in df_weekly.columns if c.startswith('total_events_W')], key=lambda x: int(x.split('_W')[1]))
    
    df_sorted = df_weekly.merge(df_feats[['User ID', 'first_active_week', 'last_active_week']], on='User ID')
    df_sorted = df_sorted.sort_values(['first_active_week', 'last_active_week'])
    
    w_data = df_sorted[evt_cols].values
    sns.heatmap(w_data, cmap='Blues', vmin=0, vmax=np.percentile(w_data, 95) if len(w_data)>0 else 100, cbar_kws={'label': 'Events'})
    plt.title('Student Weekly Activity Heatmap (IT-2507)')
    plt.xlabel('Week Number')
    plt.ylabel('Students')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'fig8_moodle_event_heatmap.png'), bbox_inches='tight')
    plt.close()
    print("Saved fig8")
except Exception as e:
    print(f"Error fig8: {e}")

print("Figure 9: Content Diversity Over Time")
try:
    plt.figure(figsize=(12, 6))
    cats = ['assessment_events_W', 'learning_events_W', 'reference_events_W', 'interactive_events_W', 'submission_events_W']
    labels = ['Assessment', 'Learning', 'Reference', 'Interactive', 'Submission']
    colors = ['orange', 'blue', 'green', 'purple', 'red']
    
    stacks = []
    weeks = []
    for cat in cats:
        cols = sorted([c for c in df_weekly.columns if c.startswith(cat)], key=lambda x: int(x.split('_W')[1]))
        stacks.append(df_weekly[cols].mean().values)
        if not weeks: weeks = range(1, len(cols) + 1)
        
    plt.stackplot(weeks, stacks, labels=labels, colors=colors, alpha=0.7)
    plt.title('Weekly Content Engagement Profile')
    plt.xlabel('Week Number')
    plt.ylabel('Mean Events')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'fig9_content_diversity_over_time.png'), bbox_inches='tight')
    plt.close()
    print("Saved fig9")
except Exception as e:
    print(f"Error fig9: {e}")

print("Figure 10: Session Patterns")
try:
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2)
    
    df_feats['submitted_str'] = df_feats['submitted_any_assignment'].map({1: 'submitted', 0: 'not_submitted'})
    
    ax0 = plt.subplot(gs[0, 0])
    sns.boxplot(x='submitted_str', y='mean_session_duration_min', data=df_feats, palette=COLOR_MAP, ax=ax0, showfliers=False)
    ax0.set_title('Mean Session Duration')
    
    # We don't have hourly/dayofweek aggregate means readily available from feats without raw log
    # Plotting distributions from features instead
    ax1 = plt.subplot(gs[0, 1])
    sns.kdeplot(df_feats['pct_evening_events'].dropna(), ax=ax1, fill=True)
    ax1.set_title('Pct Evening Events Density')
    
    ax2 = plt.subplot(gs[1, 0])
    sns.kdeplot(df_feats['pct_weekend_events'].dropna(), ax=ax2, fill=True)
    ax2.set_title('Pct Weekend Events Density')
    
    ax3 = plt.subplot(gs[1, 1])
    df_feats['engagement_str'] = df_feats['high_engagement'].map({1: 'high_engagement', 0: 'low_engagement'})
    for eng in ['high_engagement', 'low_engagement']:
        sns.kdeplot(df_feats[df_feats['engagement_str']==eng]['total_sessions'].dropna(), label=eng, color=COLOR_MAP.get(eng, 'gray'), ax=ax3, fill=True, alpha=0.2)
    ax3.set_title('Total Sessions Density by Engagement')
    ax3.legend()
    
    plt.suptitle('Session and Temporal Access Patterns (Moodle)')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'fig10_session_patterns.png'), bbox_inches='tight')
    plt.close()
    print("Saved fig10")
except Exception as e:
    print(f"Error fig10: {e}")

print("Done generating Moodle figures.")
