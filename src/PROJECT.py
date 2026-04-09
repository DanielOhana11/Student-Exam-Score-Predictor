import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

#Path definitions

TRAIN_FILE = r"C:\Users\danie\Downloads\student_train.csv"
TEST_FILE = r"C:\Users\danie\Downloads\validation_features_public.csv"


#Cleaning and filling functions
def clean_and_fill(df, train_stats=None):
    df = df.copy()
    rules = {
        'attendance_rate': (0.0, 1.0), 'midterm_score': (0, 100),
        'homework_avg_score': (0, 100), 'study_hours_per_week': (0, 168),
        'sleep_hours_per_night': (0, 24), 'time_on_social_media_min_per_day': (0, 24 * 60),
    }
    for col, (min_val, max_val) in rules.items():
        if col in df.columns:
            mask = (df[col] < min_val) | (df[col] > max_val)
            df.loc[mask, col] = np.nan

    for col in ['gender', 'has_part_time_job']:
        if col in df.columns:
            df.loc[~df[col].isin([0, 1]), col] = np.nan

    if train_stats is None:
        stats = {}
        calc_mode = True
    else:
        stats = train_stats
        calc_mode = False

    for col in df.columns:
        if col == 'final_exam_score': continue
        if df[col].dtype == 'object' or col in ['gender', 'has_part_time_job']:
            val = df[col].mode()[0] if calc_mode else stats.get(col, 0)
            if calc_mode: stats[col] = val
            df[col] = df[col].fillna(val)
        else:
            val = df[col].median() if calc_mode else stats.get(col, 0)
            if calc_mode: stats[col] = val
            df[col] = df[col].fillna(val)
    return df, stats


#Loading and preparation
print("--- Loading data... ---")
df_train = pd.read_csv(TRAIN_FILE)
df_test_file = pd.read_csv(TEST_FILE)

df_train = df_train.drop_duplicates().dropna(subset=['final_exam_score'])
df_train, train_stats = clean_and_fill(df_train)
df_test_file, _ = clean_and_fill(df_test_file, train_stats)

# Data processing for the model
df_train = pd.get_dummies(df_train, columns=['favorite_music_genre'], drop_first=True)
df_test_file = pd.get_dummies(df_test_file, columns=['favorite_music_genre'], drop_first=True)

for col in ['time_on_social_media_min_per_day', 'study_hours_per_week']:
    if col in df_train.columns:
        df_train[col] = np.log1p(df_train[col])
        df_test_file[col] = np.log1p(df_test_file[col])

#Deleting weak variables and adding squares
cols_to_drop = ['commute_time_minutes', 'has_part_time_job', 'gender']
df_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
df_test_file = df_test_file.drop(columns=[c for c in cols_to_drop if c in df_test_file.columns])

df_train['study_hours_sq'] = df_train['study_hours_per_week'] ** 2
df_test_file['study_hours_sq'] = df_test_file['study_hours_per_week'] ** 2

df_train['sleep_hours_sq'] = df_train['sleep_hours_per_night'] ** 2
df_test_file['sleep_hours_sq'] = df_test_file['sleep_hours_per_night'] ** 2

# (Attendance * Study hours)
df_train['study_attendance_interaction'] = df_train['study_hours_per_week'] * df_train['attendance_rate']
df_test_file['study_attendance_interaction'] = df_test_file['study_hours_per_week'] * df_test_file['attendance_rate']

# (Midterm exam * Homework)
df_train['midterm_homework_interaction'] = df_train['midterm_score'] * df_train['homework_avg_score']
df_test_file['midterm_homework_interaction'] = df_test_file['midterm_score'] * df_test_file['homework_avg_score']


X = df_train.drop('final_exam_score', axis=1)
y = df_train['final_exam_score']

# Column alignment
missing = set(X.columns) - set(df_test_file.columns)
for c in missing: df_test_file[c] = 0
df_test_file = df_test_file[X.columns]


# Outlier filtering and final training (Linear Regression)
print("--- Linear model trainer and outlier filter... ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(df_test_file)

# outlier filter
initial_model = LinearRegression()
initial_model.fit(X_scaled, y)
residuals = np.abs(y - initial_model.predict(X_scaled))
mask_clean = residuals < (3 * np.std(residuals))

X_cleaned = X_scaled[mask_clean]
y_cleaned = y[mask_clean]
print(f"   Deleted  {len(y) - len(y_cleaned)} Exceptions.")

# Division into Train and Validation
X_train, X_val, y_train, y_val = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# אימון סופי
final_model = LinearRegression()
final_model.fit(X_train, y_train)

# Calculating grades
train_r2 = final_model.score(X_train, y_train)
test_r2 = final_model.score(X_val, y_val)

print("\nPerformance Metrics:")
print(f"🏆 Train R²: {train_r2:.5f} (Checking compliance with learned data)")
print(f"🏆 Test R²:  {test_r2:.5f} (Checking compliance with learned data)")

if abs(train_r2 - test_r2) < 0.05:
    print("The gap between Train and Test is small - there is no significant overfitting.")
else:
    print("There is a gap between Train and Test - pay attention to Overfitting.")

# Creating a grade file
final_model.fit(X_cleaned, y_cleaned)
final_preds = final_model.predict(X_test_scaled)
final_preds = np.clip(final_preds, 0, 100)
pd.DataFrame({'final_exam_score': final_preds}).to_csv("daniel_ohana_final.csv", index=False)

# Print the equation
print("\n=== The final equation of the model ===")
print(f"Intercept (b0) = {final_model.intercept_:.4f}")
coefs = pd.DataFrame({'Feature': X.columns, 'Coefficient (Weight)': final_model.coef_})
print(coefs.sort_values(by='Coefficient (Weight)', ascending=False).to_string(index=False))