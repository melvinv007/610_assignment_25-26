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
COLOR_PALETTE = {'Distinction': '#2ecc71', 'Pass': '#3498db', 'Fail': '#e67e22', 'Withdrawn': '#e74c3c'}

out_data_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\data'
fig_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\figures'
os.makedirs(fig_path, exist_ok=True)

try:
    df_master = pd.read_csv(os.path.join(out_data_path, 'oulad_master.csv'))
    df_weekly = pd.read_csv(os.path.join(out_data_path, 'oulad_weekly_clicks.csv'))
    df_latency = pd.read_csv(os.path.join(out_data_path, 'oulad_latency.csv'))
    
    df_features = pd.read_csv(os.path.join(out_data_path, 'oulad_features_ml.csv')) if os.path.exists(os.path.join(out_data_path, 'oulad_features_ml.csv')) else df_master
except FileNotFoundError as e:
    print(f"Data files missing: {e}")
    exit(1)

print("Figure 1: Weekly Engagement")
try:
    plt.figure(figsize=(10, 6))
    pos_weeks = [c for c in df_weekly.columns if c.startswith('W') and not c.startswith('Wpre')]
    df_w_merge = df_weekly.merge(df_master[['id_student', 'code_presentation', 'final_result']], on=['id_student', 'code_presentation'])
    
    agg_w = df_w_merge.groupby('final_result')[pos_weeks].mean().T
    agg_w.index = [int(i[1:]) for i in agg_w.index]
    
    for res in ['Distinction', 'Pass', 'Fail', 'Withdrawn']:
        if res in agg_w.columns:
            plt.plot(agg_w.index, agg_w[res], label=res, color=COLOR_PALETTE[res], linewidth=2)
    
    plt.axvline(x=4, color='k', linestyle='--', alpha=0.5, label='Early Warning Window')
    plt.title('Weekly VLE Engagement Trajectories by Outcome Group')
    plt.xlabel('Week Number')
    plt.ylabel('Mean Clicks')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'fig1_weekly_engagement_by_outcome.png'), bbox_inches='tight')
    plt.close()
    print("Saved fig1")
except Exception as e:
    print(f"Error fig1: {e}")

print("Figure 2: Heatmap")
try:
    df_w_merge_sorted = df_w_merge.sort_values(['final_result', 'total_clicks'], ascending=[True, False]).head(500)
    w_data = df_w_merge_sorted[[c for c in pos_weeks if int(c[1:]) <= 20]].values
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(w_data, cmap='YlOrRd', vmin=0, vmax=np.percentile(w_data, 95), cbar_kws={'label': 'Clicks'})
    plt.title('Student-Week Click Intensity (Sorted by Outcome - Sample of 500)')
    plt.xlabel('Week Number')
    plt.ylabel('Students')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'fig2_engagement_heatmap.png'), bbox_inches='tight')
    plt.close()
    print("Saved fig2")
except Exception as e:
    print(f"Error fig2: {e}")

print("Figure 3: Latency Trajectory")
try:
    plt.figure(figsize=(8, 5))
    df_lat_full = df_latency.merge(df_master[['id_student', 'code_presentation', 'final_result']], on=['id_student', 'code_presentation'])
    
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Midpoint')
    # Can't plot per-TMA easily without recreating from raw studentAssessment. Instead plotting mean latency density
    for res in ['Distinction', 'Pass', 'Fail', 'Withdrawn']:
        if res in df_lat_full['final_result'].values:
            sns.kdeplot(df_lat_full[df_lat_full['final_result']==res]['mean_latency'].dropna(), label=res, color=COLOR_PALETTE[res], fill=True, alpha=0.1)
    
    plt.title('TMA Submission Latency Density by Outcome Group')
    plt.xlabel('Relative Latency (0 = Open, 1 = Deadline)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'fig3_latency_trajectory_by_outcome.png'), bbox_inches='tight')
    plt.close()
    print("Saved fig3")
except Exception as e:
    print(f"Error fig3: {e}")

print("Figure 4: Latency Trend Distribution")
try:
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    ax0 = plt.subplot(gs[0])
    ct = pd.crosstab(df_lat_full['final_result'], df_lat_full['latency_trend'], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, ax=ax0)
    ax0.set_title('Latency Trend % by Outcome')
    ax0.set_ylabel('Percentage')
    
    ax1 = plt.subplot(gs[1])
    sns.boxplot(x='final_result', y='latency_slope', data=df_lat_full, palette=COLOR_PALETTE, ax=ax1, showfliers=False)
    ax1.set_title('Latency Slope Distribution')
    
    plt.suptitle('H1: Submission Latency Trend by Outcome')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'fig4_latency_trend_distribution.png'), bbox_inches='tight')
    plt.close()
    print("Saved fig4")
except Exception as e:
    print(f"Error fig4: {e}")

print("Figure 5: Temporal Engagement Features")
try:
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2)
    
    ax0 = plt.subplot(gs[0, 0])
    sns.violinplot(x='final_result', y='early_ratio', data=df_w_merge, palette=COLOR_PALETTE, ax=ax0)
    ax0.set_title('Early Ratio by Outcome')
    
    ax1 = plt.subplot(gs[0, 1])
    sns.violinplot(x='final_result', y='late_clicks', data=df_w_merge, palette=COLOR_PALETTE, ax=ax1)
    ax1.set_title('Late Clicks by Outcome')
    
    ax2 = plt.subplot(gs[1, 0])
    sns.scatterplot(x='early_ratio', y='total_clicks', hue='final_result', data=df_w_merge.sample(1000), palette=COLOR_PALETTE, ax=ax2, alpha=0.5)
    ax2.set_title('Total Clicks vs Early Ratio')
    
    ax3 = plt.subplot(gs[1, 1])
    for res in ['Distinction', 'Pass', 'Fail', 'Withdrawn']:
        sns.kdeplot(df_w_merge[df_w_merge['final_result']==res]['regularity'].dropna(), label=res, color=COLOR_PALETTE[res], ax=ax3, fill=True, alpha=0.1)
    ax3.set_title('Regularity Density')
    ax3.legend()
    
    plt.suptitle('Temporal Engagement Feature Distributions by Outcome')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'fig5_early_ratio_vs_outcome.png'), bbox_inches='tight')
    plt.close()
    print("Saved fig5")
except Exception as e:
    print(f"Error fig5: {e}")

print("Done generating OULAD figures.")
