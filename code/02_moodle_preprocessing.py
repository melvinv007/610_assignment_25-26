import pandas as pd
import numpy as np
import scipy.stats
import os

filepath = r'C:\Users\vmelv\Downloads\610_assignment\dataset_2.csv'
out_data_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\data'
out_res_path = r'C:\Users\vmelv\Downloads\610_assignment\outputs\results'
os.makedirs(out_data_path, exist_ok=True)
os.makedirs(out_res_path, exist_ok=True)

print("1. Loading and cleaning Moodle data")
df = pd.read_csv(filepath, encoding='utf-8-sig')
df['User ID'] = df['User ID'].astype(str).str.strip().str.strip("'")
df['Time'] = pd.to_datetime(df['Time'], format='%d/%m/%y, %H:%M:%S', errors='coerce')

df = df.dropna(subset=['User ID', 'Time'])
df = df[df['User ID'].str.isnumeric()] # Only keep student-like IDs if needed, assuming numeric
df = df.sort_values(['User ID', 'Time'])
print(f"Shape: {df.shape}, Unique Users: {df['User ID'].nunique()}")

print("2. Course week assignment")
course_start = df['Time'].min()
df['week_number'] = ((df['Time'] - course_start).dt.days // 7) + 1
df = df[df['week_number'] <= 40] # Clip to 40 weeks
df['date'] = df['Time'].dt.date
df['hour'] = df['Time'].dt.hour
df['day_of_week'] = df['Time'].dt.dayofweek

print("3. Content category classification")
def map_content(code):
    if pd.isna(code): return 'Other'
    code = str(code)
    if code in ['Continuous Evaluation', 'Practice Quiz']: return 'Assessment'
    if code in ['Must Know Concepts Videos', 'Self-Learning Material', 'Audiobyte', 'Faculty Video', 'Live Session Recordings', 'Class PPT']: return 'Learning_Content'
    if code in ['Additional Learning Material', 'Reference Material', 'Glossary']: return 'Reference'
    if code in ['Digital Storyboard', 'Discussion Forum']: return 'Interactive'
    if code in ['Course Project', 'Assignment Submissions']: return 'Submission'
    return 'Other'

df['content_category'] = df['Content code'].apply(map_content)
df['is_quiz_attempt'] = df['Event name'].isin(['Quiz attempt started', 'Quiz attempt submitted', 'Quiz attempt updated', 'Quiz attempt reviewed']).astype(int)
df['is_quiz_submit'] = (df['Event name'] == 'Quiz attempt submitted').astype(int)
df['is_assignment_submit'] = df['Event name'].isin(['A submission has been submitted.', 'Submission created.']).astype(int)

df.to_csv(os.path.join(out_data_path, 'moodle_clean.csv'), index=False)

print("4. Sessionization")
df['time_gap'] = df.groupby('User ID')['Time'].diff().dt.total_seconds() / 60.0
df['new_session'] = (df['time_gap'].isna()) | (df['time_gap'] > 30)
df['session_id_part'] = df.groupby('User ID')['new_session'].cumsum()
df['session_id'] = df['User ID'].astype(str) + '_' + df['session_id_part'].astype(str)

session_stats = df.groupby('session_id').agg(
    User_ID=('User ID', 'first'),
    session_start=('Time', 'min'),
    session_end=('Time', 'max'),
    events_in_session=('Time', 'count')
).reset_index()
session_stats['session_duration_min'] = (session_stats['session_end'] - session_stats['session_start']).dt.total_seconds() / 60.0
session_stats.to_csv(os.path.join(out_data_path, 'moodle_sessions.csv'), index=False)

print("5. Weekly aggregates")
weekly = df.groupby(['User ID', 'week_number']).agg(
    total_events=('Time', 'count'),
    assessment_events=('content_category', lambda x: (x == 'Assessment').sum()),
    learning_events=('content_category', lambda x: (x == 'Learning_Content').sum()),
    reference_events=('content_category', lambda x: (x == 'Reference').sum()),
    interactive_events=('content_category', lambda x: (x == 'Interactive').sum()),
    submission_events=('content_category', lambda x: (x == 'Submission').sum()),
    quiz_attempts=('is_quiz_attempt', 'sum'),
    quiz_submits=('is_quiz_submit', 'sum'),
    assignment_submits=('is_assignment_submit', 'sum')
).reset_index()

weekly_wide = weekly.pivot_table(index='User ID', columns='week_number', fill_value=0)
weekly_wide.columns = [f"{col[0]}_W{col[1]}" for col in weekly_wide.columns]
weekly_wide = weekly_wide.reset_index()
weekly_wide.to_csv(os.path.join(out_data_path, 'moodle_weekly.csv'), index=False)

print("6. Per-student features")
student_feats = df.groupby('User ID').agg(
    total_events=('Time', 'count'),
    unique_days_active=('date', 'nunique'),
    unique_weeks_active=('week_number', 'nunique'),
    total_quiz_attempts=('is_quiz_attempt', 'sum'),
    total_quiz_submits=('is_quiz_submit', 'sum'),
    total_assignment_submits=('is_assignment_submit', 'sum'),
    first_active_week=('week_number', 'min'),
    last_active_week=('week_number', 'max'),
    pct_evening_events=('hour', lambda x: (x >= 18).mean()),
    pct_weekend_events=('day_of_week', lambda x: (x >= 5).mean())
).reset_index()

# Entropy
cat_counts = df.groupby(['User ID', 'content_category']).size().unstack(fill_value=0)
cat_probs = cat_counts.div(cat_counts.sum(axis=1), axis=0)
student_feats['content_diversity_entropy'] = -cat_probs.multiply(np.log2(cat_probs.replace(0, 1))).sum(axis=1).values

# Sessions merge
sess_agg = session_stats.groupby('User_ID').agg(
    total_sessions=('session_id', 'count'),
    mean_session_duration_min=('session_duration_min', 'mean'),
    mean_events_per_session=('events_in_session', 'mean')
).reset_index().rename(columns={'User_ID': 'User ID'})
student_feats = student_feats.merge(sess_agg, on='User ID', how='left')

# TDI
quiz_df = df[df['is_quiz_attempt'] == 1]
q_stats = quiz_df.groupby('User ID').agg(
    quiz_days=('date', 'nunique'),
    quiz_span_days=('date', lambda x: (x.max() - x.min()).days)
).reset_index()
student_feats = student_feats.merge(q_stats, on='User ID', how='left')
student_feats['TDI'] = np.where(student_feats['total_quiz_attempts'] >= 5, 
                                student_feats['quiz_days'] / np.maximum(student_feats['quiz_span_days'], 1), 
                                np.nan)

student_feats['assessment_submission_rate'] = student_feats['total_quiz_submits'] / np.maximum(student_feats['total_quiz_attempts'], 1)
student_feats['submitted_any_assignment'] = (student_feats['total_assignment_submits'] >= 1).astype(int)
student_feats['high_engagement'] = ((student_feats['submitted_any_assignment'] == 1) & 
                                    (student_feats['total_quiz_submits'] >= 1) & 
                                    (student_feats['unique_weeks_active'] >= 4)).astype(int)

student_feats.to_csv(os.path.join(out_data_path, 'moodle_student_features.csv'), index=False)

print("7. Generating Report")
with open(os.path.join(out_res_path, 'moodle_preprocessing_report.txt'), 'w') as f:
    f.write("Moodle Preprocessing Report\n===========================\n")
    f.write(f"Cleaned Shape: {df.shape}\n")
    f.write(f"Unique Users: {df['User ID'].nunique()}\n")
    f.write(f"Date Range: {df['date'].min()} to {df['date'].max()}\n\n")
    f.write("High Engagement Distribution:\n")
    f.write(student_feats['high_engagement'].value_counts().to_string() + "\n")

print("Done! Moodle preprocessing complete.")
