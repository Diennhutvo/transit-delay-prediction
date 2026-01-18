"""
Question 7: Baseline Model (Without Machine Learning)

This script implements simple baseline predictors to serve as a performance benchmark.
The goal is to establish a minimum performance level that ML models must exceed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("QUESTION 7: BASELINE MODEL (WITHOUT ML)")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")

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
data['is_weekend'] = data['day_of_week'] >= 5
data['is_rush_hour'] = data['hour'].isin([7, 8, 9, 16, 17, 18])

print(f"Total records: {len(data):,}")
print(f"Major delays (≥10 min): {data['delay_10plus'].sum():,} ({data['delay_10plus'].sum()/len(data)*100:.2f}%)")
print(f"No major delay: {(~data['delay_10plus']).sum():,} ({(~data['delay_10plus']).sum()/len(data)*100:.2f}%)")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================
print("\n[2/5] Creating train/test split...")

# Use 80% for training, 20% for testing
# Sort by timestamp to avoid data leakage
data = data.sort_values('timestamp').reset_index(drop=True)
split_idx = int(len(data) * 0.8)

train_data = data.iloc[:split_idx].copy()
test_data = data.iloc[split_idx:].copy()

print(f"Training set: {len(train_data):,} records")
print(f"Test set: {len(test_data):,} records")
print(f"Test set delay rate: {test_data['delay_10plus'].sum()/len(test_data)*100:.2f}%")

# ============================================================================
# BASELINE 1: MAJORITY CLASS (Always predict "No Delay")
# ============================================================================
print("\n[3/5] Baseline 1: Majority Class Predictor")
print("-" * 70)

# Always predict False (no major delay)
baseline1_pred = np.zeros(len(test_data), dtype=bool)
y_true = test_data['delay_10plus'].values

# Calculate metrics
baseline1_accuracy = accuracy_score(y_true, baseline1_pred)
baseline1_precision = precision_score(y_true, baseline1_pred, zero_division=0)
baseline1_recall = recall_score(y_true, baseline1_pred, zero_division=0)
baseline1_f1 = f1_score(y_true, baseline1_pred, zero_division=0)

print("Strategy: Always predict 'No Major Delay'")
print(f"\nResults:")
print(f"  Accuracy:  {baseline1_accuracy:.4f} ({baseline1_accuracy*100:.2f}%)")
print(f"  Precision: {baseline1_precision:.4f}")
print(f"  Recall:    {baseline1_recall:.4f}")
print(f"  F1 Score:  {baseline1_f1:.4f}")

print("\nConfusion Matrix:")
cm1 = confusion_matrix(y_true, baseline1_pred)
print(f"  True Negatives:  {cm1[0,0]:,}")
print(f"  False Positives: {cm1[0,1]:,}")
print(f"  False Negatives: {cm1[1,0]:,}")
print(f"  True Positives:  {cm1[1,1]:,}")

print("\n⚠️  Problem: This baseline has 0% recall - it never predicts delays!")
print("    While accuracy is high, it's useless for identifying actual delays.")

# ============================================================================
# BASELINE 2: ROUTE-BASED PREDICTOR
# ============================================================================
print("\n[4/5] Baseline 2: Route-Based Predictor")
print("-" * 70)

# Calculate historical delay rate for each route
route_delay_rates = train_data.groupby('route_id')['delay_10plus'].mean()

# Set threshold: predict delay if route's historical delay rate > 5%
threshold = 0.05
high_delay_routes = route_delay_rates[route_delay_rates > threshold].index

print(f"Strategy: Predict delay if route has historical delay rate > {threshold*100:.0f}%")
print(f"High-delay routes identified: {len(high_delay_routes)}")

# Make predictions
baseline2_pred = test_data['route_id'].isin(high_delay_routes).values

# Calculate metrics
baseline2_accuracy = accuracy_score(y_true, baseline2_pred)
baseline2_precision = precision_score(y_true, baseline2_pred, zero_division=0)
baseline2_recall = recall_score(y_true, baseline2_pred, zero_division=0)
baseline2_f1 = f1_score(y_true, baseline2_pred, zero_division=0)

print(f"\nResults:")
print(f"  Accuracy:  {baseline2_accuracy:.4f} ({baseline2_accuracy*100:.2f}%)")
print(f"  Precision: {baseline2_precision:.4f}")
print(f"  Recall:    {baseline2_recall:.4f}")
print(f"  F1 Score:  {baseline2_f1:.4f}")

print("\nConfusion Matrix:")
cm2 = confusion_matrix(y_true, baseline2_pred)
print(f"  True Negatives:  {cm2[0,0]:,}")
print(f"  False Positives: {cm2[0,1]:,}")
print(f"  False Negatives: {cm2[1,0]:,}")
print(f"  True Positives:  {cm2[1,1]:,}")

# ============================================================================
# BASELINE 3: TIME-BASED PREDICTOR
# ============================================================================
print("\n[5/5] Baseline 3: Time-Based Predictor (Rush Hour)")
print("-" * 70)

# Calculate delay rate during rush hour in training data
rush_hour_delay_rate = train_data[train_data['is_rush_hour']]['delay_10plus'].mean()
non_rush_delay_rate = train_data[~train_data['is_rush_hour']]['delay_10plus'].mean()

print(f"Rush hour delay rate (training): {rush_hour_delay_rate*100:.2f}%")
print(f"Non-rush hour delay rate (training): {non_rush_delay_rate*100:.2f}%")

# Strategy: Predict delay during rush hour if delay rate is higher
print(f"\nStrategy: Predict delay during rush hours (7-9 AM, 4-6 PM)")

# Make predictions
baseline3_pred = test_data['is_rush_hour'].values

# Calculate metrics
baseline3_accuracy = accuracy_score(y_true, baseline3_pred)
baseline3_precision = precision_score(y_true, baseline3_pred, zero_division=0)
baseline3_recall = recall_score(y_true, baseline3_pred, zero_division=0)
baseline3_f1 = f1_score(y_true, baseline3_pred, zero_division=0)

print(f"\nResults:")
print(f"  Accuracy:  {baseline3_accuracy:.4f} ({baseline3_accuracy*100:.2f}%)")
print(f"  Precision: {baseline3_precision:.4f}")
print(f"  Recall:    {baseline3_recall:.4f}")
print(f"  F1 Score:  {baseline3_f1:.4f}")

print("\nConfusion Matrix:")
cm3 = confusion_matrix(y_true, baseline3_pred)
print(f"  True Negatives:  {cm3[0,0]:,}")
print(f"  False Positives: {cm3[0,1]:,}")
print(f"  False Negatives: {cm3[1,0]:,}")
print(f"  True Positives:  {cm3[1,1]:,}")

# ============================================================================
# COMPARISON & VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("BASELINE COMPARISON")
print("="*70)

# Create comparison dataframe
results = pd.DataFrame({
    'Baseline': ['Majority Class', 'Route-Based', 'Time-Based (Rush Hour)'],
    'Accuracy': [baseline1_accuracy, baseline2_accuracy, baseline3_accuracy],
    'Precision': [baseline1_precision, baseline2_precision, baseline3_precision],
    'Recall': [baseline1_recall, baseline2_recall, baseline3_recall],
    'F1 Score': [baseline1_f1, baseline2_f1, baseline3_f1]
})

print("\n" + results.to_string(index=False))

# Find best baseline
best_f1_idx = results['F1 Score'].idxmax()
best_baseline = results.iloc[best_f1_idx]

print(f"\n🏆 Best Baseline: {best_baseline['Baseline']}")
print(f"   F1 Score: {best_baseline['F1 Score']:.4f}")

# Save results
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
results.to_csv(output_dir / "baseline_comparison.csv", index=False)
print(f"\nResults saved to: {output_dir / 'baseline_comparison.csv'}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\nCreating visualizations...")

# Create visualization directory
viz_dir = Path("visualizations")
viz_dir.mkdir(exist_ok=True)

# Plot 1: Metrics Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors = ['#3498db', '#e74c3c', '#2ecc71']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(results['Baseline'], results[metric], color=colors, edgecolor='black', alpha=0.7)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(results['Baseline'], rotation=15, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

plt.suptitle('Baseline Model Comparison (Question 7)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(viz_dir / "9_baseline_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {viz_dir / '9_baseline_comparison.png'}")
plt.close()

# Plot 2: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

cms = [cm1, cm2, cm3]
titles = ['Majority Class', 'Route-Based', 'Time-Based']

for idx, (cm, title) in enumerate(zip(cms, titles)):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                cbar=False, square=True, linewidths=1, linecolor='black')
    axes[idx].set_title(f'{title}\nConfusion Matrix', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_xticklabels(['No Delay', 'Delay'])
    axes[idx].set_yticklabels(['No Delay', 'Delay'])

plt.suptitle('Baseline Model Confusion Matrices', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(viz_dir / "10_baseline_confusion_matrices.png", dpi=300, bbox_inches='tight')
print(f"Saved: {viz_dir / '10_baseline_confusion_matrices.png'}")
plt.close()

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("SUMMARY & RECOMMENDATIONS")
print("="*70)

print(f"""
Key Findings:

1. MAJORITY CLASS BASELINE
   • Achieves high accuracy ({baseline1_accuracy*100:.1f}%) but is USELESS
   • Has 0% recall - never identifies actual delays
   • Demonstrates why accuracy alone is misleading with imbalanced data

2. ROUTE-BASED BASELINE
   • Accuracy: {baseline2_accuracy*100:.1f}%
   • Recall: {baseline2_recall*100:.1f}%
   • F1 Score: {baseline2_f1:.4f}
   • Shows that route information has some predictive value

3. TIME-BASED BASELINE (RUSH HOUR)
   • Accuracy: {baseline3_accuracy*100:.1f}%
   • Recall: {baseline3_recall*100:.1f}%
   • F1 Score: {baseline3_f1:.4f}
   • Temporal patterns exist but are not strong predictors alone

BEST BASELINE: {best_baseline['Baseline']}
   • F1 Score: {best_baseline['F1 Score']:.4f}
   • This is the benchmark for Question 8 (Logistic Regression)

For Question 8:
   ✓ Your logistic regression model should achieve F1 > {best_baseline['F1 Score']:.4f}
   ✓ Focus on improving recall (identifying actual delays)
   ✓ Consider combining route + time features + additional predictors
   ✓ Address class imbalance with class_weight='balanced'
""")

print("="*70)
print("QUESTION 7 COMPLETE!")
print("="*70)
print(f"\nNext step: Proceed to Question 8 (Logistic Regression)")
print(f"Goal: Beat F1 score of {best_baseline['F1 Score']:.4f}")
