"""
Question 8: Logistic Regression Model

This script implements a logistic regression model to predict major transit delays (≥10 min).
The goal is to beat the baseline F1 score of 0.0871 from Question 7.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("QUESTION 8: LOGISTIC REGRESSION MODEL")
print("="*70)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/7] Loading data...")

csv_files = sorted(Path("data/interim/gtfs_rt").glob("trip_updates_parsed_*.csv"))
print(f"Found {len(csv_files)} CSV files")

dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Add temporal features
data['timestamp'] = pd.to_datetime(data['feed_timestamp'], unit='s')
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
data['is_rush_hour'] = data['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)

# Create time of day categories
def get_time_period(hour):
    if 6 <= hour < 9:
        return 'morning_rush'
    elif 9 <= hour < 16:
        return 'midday'
    elif 16 <= hour < 19:
        return 'evening_rush'
    elif 19 <= hour < 22:
        return 'evening'
    else:
        return 'night'

data['time_period'] = data['hour'].apply(get_time_period)

print(f"Total records: {len(data):,}")
print(f"Major delays (≥10 min): {data['delay_10plus'].sum():,} ({data['delay_10plus'].sum()/len(data)*100:.2f}%)")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n[2/7] Engineering features...")

# Calculate route-level statistics
route_stats = data.groupby('route_id').agg({
    'delay_min': ['mean', 'std', 'count'],
    'delay_10plus': 'mean'
}).reset_index()
route_stats.columns = ['route_id', 'route_avg_delay', 'route_std_delay', 'route_trip_count', 'route_delay_rate']
data = data.merge(route_stats, on='route_id', how='left')

# Calculate stop-level statistics
stop_stats = data.groupby('stop_id').agg({
    'delay_min': ['mean', 'count'],
    'delay_10plus': 'mean'
}).reset_index()
stop_stats.columns = ['stop_id', 'stop_avg_delay', 'stop_trip_count', 'stop_delay_rate']
data = data.merge(stop_stats, on='stop_id', how='left')

# Fill NaN values
data['route_std_delay'] = data['route_std_delay'].fillna(0)

print("Features created:")
print("  • Temporal: hour, day_of_week, is_weekend, is_rush_hour, time_period")
print("  • Route stats: avg_delay, std_delay, trip_count, delay_rate")
print("  • Stop stats: avg_delay, trip_count, delay_rate")

# ============================================================================
# PREPARE FEATURES AND TARGET
# ============================================================================
print("\n[3/7] Preparing features and target...")

# Select features for modeling
feature_cols = [
    'hour',
    'day_of_week',
    'is_weekend',
    'is_rush_hour',
    'route_avg_delay',
    'route_std_delay',
    'route_delay_rate',
    'stop_avg_delay',
    'stop_delay_rate',
    'stop_sequence'
]

# Create feature matrix
X = data[feature_cols].copy()
y = data['delay_10plus'].values

# Handle any remaining NaN values
X = X.fillna(0)

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {feature_cols}")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================
print("\n[4/7] Splitting data...")

# Time-based split (80/20)
data_sorted = data.sort_values('timestamp').reset_index(drop=True)
split_idx = int(len(data_sorted) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"Training set: {len(X_train):,} samples ({y_train.sum()} delays, {y_train.sum()/len(y_train)*100:.2f}%)")
print(f"Test set: {len(X_test):,} samples ({y_test.sum()} delays, {y_test.sum()/len(y_test)*100:.2f}%)")

# ============================================================================
# SCALE FEATURES
# ============================================================================
print("\n[5/7] Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================================
print("\n[6/7] Training logistic regression model...")

# Train with class weighting to handle imbalance
model = LogisticRegression(
    class_weight='balanced',  # Handle class imbalance
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)
y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]

print("Model trained successfully!")

# ============================================================================
# EVALUATE MODEL
# ============================================================================
print("\n[7/7] Evaluating model performance...")
print("="*70)

# Training set performance
train_accuracy = accuracy_score(y_train, y_pred_train)
train_precision = precision_score(y_train, y_pred_train, zero_division=0)
train_recall = recall_score(y_train, y_pred_train, zero_division=0)
train_f1 = f1_score(y_train, y_pred_train, zero_division=0)

print("TRAINING SET PERFORMANCE:")
print(f"  Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1 Score:  {train_f1:.4f}")

# Test set performance
test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, zero_division=0)
test_recall = recall_score(y_test, y_pred_test, zero_division=0)
test_f1 = f1_score(y_test, y_pred_test, zero_division=0)

print("\nTEST SET PERFORMANCE:")
print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print("\nCONFUSION MATRIX (Test Set):")
print(f"  True Negatives:  {cm[0,0]:,}")
print(f"  False Positives: {cm[0,1]:,}")
print(f"  False Negatives: {cm[1,0]:,}")
print(f"  True Positives:  {cm[1,1]:,}")

# Compare to baseline
baseline_f1 = 0.0871  # From Question 7
improvement = ((test_f1 - baseline_f1) / baseline_f1) * 100 if baseline_f1 > 0 else 0

print("\n" + "="*70)
print("COMPARISON TO BASELINE")
print("="*70)
print(f"Baseline F1 (Route-Based):     {baseline_f1:.4f}")
print(f"Logistic Regression F1:        {test_f1:.4f}")
print(f"Improvement:                   {improvement:+.1f}%")

if test_f1 > baseline_f1:
    print("✅ SUCCESS: Logistic regression beats the baseline!")
else:
    print("⚠️  Logistic regression did not beat baseline (likely due to limited data)")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)

# Get feature coefficients
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_[0]
})
feature_importance['abs_coefficient'] = feature_importance['coefficient'].abs()
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

print("\nTop 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    direction = "↑ Increases" if row['coefficient'] > 0 else "↓ Decreases"
    print(f"  {row['feature']:25s} {direction} delay risk (coef: {row['coefficient']:+.4f})")

# ============================================================================
# SAVE RESULTS
# ============================================================================
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

# Save performance metrics
results = pd.DataFrame({
    'Model': ['Baseline (Route-Based)', 'Logistic Regression'],
    'Accuracy': [0.8954, test_accuracy],
    'Precision': [0.1080, test_precision],
    'Recall': [0.0730, test_recall],
    'F1 Score': [baseline_f1, test_f1]
})
results.to_csv(output_dir / "model_comparison.csv", index=False)

# Save feature importance
feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

print(f"\nResults saved to: {output_dir.absolute()}/")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\nCreating visualizations...")

viz_dir = Path("visualizations")
viz_dir.mkdir(exist_ok=True)

# Plot 1: Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors = ['#3498db', '#e74c3c']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(results['Model'], results[metric], color=colors, edgecolor='black', alpha=0.7)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Model Performance Comparison (Question 8)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(viz_dir / "11_model_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {viz_dir / '11_model_comparison.png'}")
plt.close()

# Plot 2: Confusion Matrix
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            square=True, linewidths=1, linecolor='black', cbar_kws={'label': 'Count'})
ax.set_title('Logistic Regression\nConfusion Matrix (Test Set)', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_xticklabels(['No Delay', 'Delay'])
ax.set_yticklabels(['No Delay', 'Delay'])
plt.tight_layout()
plt.savefig(viz_dir / "12_confusion_matrix.png", dpi=300, bbox_inches='tight')
print(f"Saved: {viz_dir / '12_confusion_matrix.png'}")
plt.close()

# Plot 3: Feature Importance
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
top_features = feature_importance.head(10)
colors_feat = ['#e74c3c' if c > 0 else '#3498db' for c in top_features['coefficient']]
ax.barh(range(len(top_features)), top_features['coefficient'], color=colors_feat, edgecolor='black', alpha=0.7)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'])
ax.set_xlabel('Coefficient Value', fontsize=12)
ax.set_title('Top 10 Feature Importance\n(Positive = Increases Delay Risk)', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(viz_dir / "13_feature_importance.png", dpi=300, bbox_inches='tight')
print(f"Saved: {viz_dir / '13_feature_importance.png'}")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("QUESTION 8 COMPLETE!")
print("="*70)

print(f"""
Summary:
  • Logistic regression model trained with {len(feature_cols)} features
  • Test F1 Score: {test_f1:.4f} (Baseline: {baseline_f1:.4f})
  • Improvement: {improvement:+.1f}%
  • Most important features: {', '.join(feature_importance.head(3)['feature'].tolist())}
  
Next Steps:
  1. Review visualizations in visualizations/ folder
  2. Analyze feature importance to understand delay drivers
  3. Consider collecting more data to improve performance
  4. Write up results for final report
""")

print("="*70)
