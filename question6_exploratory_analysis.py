"""
Question 6: Plot distributions of features and outcomes
This script creates visualizations for the transit delay prediction project
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the parsed realtime data
csv_files = sorted(Path("data/interim/gtfs_rt").glob("trip_updates_parsed_*.csv"))
print(f"Found {len(csv_files)} CSV files")

# Load all data
dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"\nTotal records: {len(data)}")
print(f"Unique routes: {data['route_id'].nunique()}")
print(f"Unique trips: {data['trip_id'].nunique()}")
print(f"Unique stops: {data['stop_id'].nunique()}")

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['feed_timestamp'], unit='s')
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['day_name'] = data['timestamp'].dt.day_name()

# Create outcome variable
data['major_delay'] = data['delay_min'] >= 10  # Changed from 15 to 10 minutes

print("\n" + "="*60)
print("CREATING VISUALIZATIONS FOR QUESTION 6")
print("="*60)

# Create output directory
output_dir = Path("visualizations")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# PLOT 1: Delay Distribution (Histogram)
# ============================================================================
print("\n[1/8] Creating delay distribution histogram...")
plt.figure(figsize=(12, 6))
plt.hist(data['delay_min'], bins=100, range=(-30, 60), edgecolor='black', alpha=0.7)
plt.axvline(x=10, color='red', linestyle='--', linewidth=2, label='Major Delay Threshold (10 min)')
plt.axvline(x=0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='On Time')
plt.xlabel('Delay (minutes)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Transit Delays', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "1_delay_distribution.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '1_delay_distribution.png'}")
plt.close()

# ============================================================================
# PLOT 2: Outcome Variable (Binary Classification)
# ============================================================================
print("[2/8] Creating outcome variable bar chart...")
plt.figure(figsize=(10, 6))
outcome_counts = data['major_delay'].value_counts()
colors = ['#2ecc71', '#e74c3c']  # Green for no delay, red for delay
bars = plt.bar(['No Major Delay\n(< 10 min)', 'Major Delay\n(≥ 10 min)'], 
               outcome_counts.values, color=colors, edgecolor='black', linewidth=1.5)

# Add percentages on bars
total = len(data)
for i, bar in enumerate(bars):
    height = bar.get_height()
    percentage = (height / total) * 100
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}\n({percentage:.1f}%)',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel('Number of Observations', fontsize=12)
plt.title('Outcome Variable Distribution: Major Delays (≥10 min)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / "2_outcome_variable.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '2_outcome_variable.png'}")
plt.close()

# ============================================================================
# PLOT 3: Delays by Hour of Day
# ============================================================================
print("[3/8] Creating delays by hour of day...")
plt.figure(figsize=(14, 6))
sns.boxplot(data=data, x='hour', y='delay_min', palette='coolwarm')
plt.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Major Delay Threshold')
plt.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='On Time')
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Delay (minutes)', fontsize=12)
plt.title('Delay Distribution by Hour of Day', fontsize=14, fontweight='bold')
plt.legend()
plt.ylim(-30, 60)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / "3_delays_by_hour.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '3_delays_by_hour.png'}")
plt.close()

# ============================================================================
# PLOT 4: Major Delay Rate by Hour
# ============================================================================
print("[4/8] Creating major delay rate by hour...")
hourly_delay_rate = data.groupby('hour')['major_delay'].agg(['sum', 'count', 'mean'])
hourly_delay_rate['rate'] = hourly_delay_rate['mean'] * 100

plt.figure(figsize=(14, 6))
bars = plt.bar(hourly_delay_rate.index, hourly_delay_rate['rate'], 
               color='#e74c3c', edgecolor='black', alpha=0.7)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Major Delay Rate (%)', fontsize=12)
plt.title('Percentage of Trips with Major Delays (≥10 min) by Hour', fontsize=14, fontweight='bold')
plt.xticks(range(24))
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / "4_delay_rate_by_hour.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '4_delay_rate_by_hour.png'}")
plt.close()

# ============================================================================
# PLOT 5: Top 15 Routes by Volume
# ============================================================================
print("[5/8] Creating top routes by volume...")
top_routes = data['route_id'].value_counts().head(15)

plt.figure(figsize=(12, 8))
plt.barh(range(len(top_routes)), top_routes.values, color='#3498db', edgecolor='black')
plt.yticks(range(len(top_routes)), [f"Route {int(r)}" for r in top_routes.index])
plt.xlabel('Number of Observations', fontsize=12)
plt.ylabel('Route', fontsize=12)
plt.title('Top 15 Routes by Number of Observations', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(output_dir / "5_top_routes_volume.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '5_top_routes_volume.png'}")
plt.close()

# ============================================================================
# PLOT 6: Delay Rate by Top Routes
# ============================================================================
print("[6/8] Creating delay rate by top routes...")
# Get top 15 routes by volume
top_route_ids = data['route_id'].value_counts().head(15).index
top_route_data = data[data['route_id'].isin(top_route_ids)]

route_delay_rate = top_route_data.groupby('route_id')['major_delay'].agg(['sum', 'count', 'mean'])
route_delay_rate['rate'] = route_delay_rate['mean'] * 100
route_delay_rate = route_delay_rate.sort_values('rate', ascending=True)

plt.figure(figsize=(12, 8))
colors = ['#e74c3c' if r > 5 else '#3498db' for r in route_delay_rate['rate']]
plt.barh(range(len(route_delay_rate)), route_delay_rate['rate'], color=colors, edgecolor='black')
plt.yticks(range(len(route_delay_rate)), [f"Route {int(r)}" for r in route_delay_rate.index])
plt.xlabel('Major Delay Rate (%)', fontsize=12)
plt.ylabel('Route', fontsize=12)
plt.title('Major Delay Rate by Top 15 Routes (by volume)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(output_dir / "6_delay_rate_by_route.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '6_delay_rate_by_route.png'}")
plt.close()

# ============================================================================
# PLOT 7: Delay Statistics Summary
# ============================================================================
print("[7/8] Creating delay statistics summary...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Delay distribution (zoomed)
axes[0, 0].hist(data['delay_min'], bins=50, range=(-10, 30), edgecolor='black', alpha=0.7, color='#3498db')
axes[0, 0].axvline(x=10, color='red', linestyle='--', linewidth=2, label='Threshold')
axes[0, 0].set_xlabel('Delay (minutes)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Delay Distribution (Zoomed: -10 to 30 min)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Mean delay by hour
hourly_mean = data.groupby('hour')['delay_min'].mean()
axes[0, 1].plot(hourly_mean.index, hourly_mean.values, marker='o', linewidth=2, markersize=8, color='#e74c3c')
axes[0, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('Mean Delay (minutes)')
axes[0, 1].set_title('Average Delay by Hour')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(range(24))

# Subplot 3: Cumulative distribution
sorted_delays = np.sort(data['delay_min'].dropna())
cumulative = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays) * 100
axes[1, 0].plot(sorted_delays, cumulative, linewidth=2, color='#9b59b6')
axes[1, 0].axvline(x=10, color='red', linestyle='--', linewidth=2, label='10 min threshold')
axes[1, 0].axhline(y=50, color='gray', linestyle=':', alpha=0.5)
axes[1, 0].set_xlabel('Delay (minutes)')
axes[1, 0].set_ylabel('Cumulative Percentage (%)')
axes[1, 0].set_title('Cumulative Distribution of Delays')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(-30, 60)

# Subplot 4: Summary statistics table
stats_text = f"""
DELAY STATISTICS SUMMARY

Total Observations: {len(data):,}
Unique Routes: {data['route_id'].nunique()}
Unique Trips: {data['trip_id'].nunique()}
Unique Stops: {data['stop_id'].nunique()}

DELAY METRICS:
Mean Delay: {data['delay_min'].mean():.2f} min
Median Delay: {data['delay_min'].median():.2f} min
Std Dev: {data['delay_min'].std():.2f} min
Min Delay: {data['delay_min'].min():.2f} min
Max Delay: {data['delay_min'].max():.2f} min

OUTCOME VARIABLE:
Major Delays (≥10 min): {data['major_delay'].sum():,} ({data['major_delay'].sum()/len(data)*100:.2f}%)
No Major Delay: {(~data['major_delay']).sum():,} ({(~data['major_delay']).sum()/len(data)*100:.2f}%)

CLASS IMBALANCE RATIO: {(~data['major_delay']).sum() / data['major_delay'].sum():.1f}:1
"""
axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace', 
                verticalalignment='center', transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')

plt.suptitle('Transit Delay Analysis - Summary Statistics', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / "7_summary_statistics.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '7_summary_statistics.png'}")
plt.close()

# ============================================================================
# PLOT 8: Day of Week Analysis
# ============================================================================
print("[8/8] Creating day of week analysis...")
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_delay_rate = data.groupby('day_name')['major_delay'].agg(['sum', 'count', 'mean'])
day_delay_rate['rate'] = day_delay_rate['mean'] * 100
day_delay_rate = day_delay_rate.reindex(day_order)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Observations by day
axes[0].bar(range(len(day_delay_rate)), day_delay_rate['count'], 
            color='#3498db', edgecolor='black', alpha=0.7)
axes[0].set_xticks(range(len(day_order)))
axes[0].set_xticklabels(day_order, rotation=45, ha='right')
axes[0].set_ylabel('Number of Observations')
axes[0].set_title('Observations by Day of Week')
axes[0].grid(True, alpha=0.3, axis='y')

# Subplot 2: Delay rate by day
axes[1].bar(range(len(day_delay_rate)), day_delay_rate['rate'], 
            color='#e74c3c', edgecolor='black', alpha=0.7)
axes[1].set_xticks(range(len(day_order)))
axes[1].set_xticklabels(day_order, rotation=45, ha='right')
axes[1].set_ylabel('Major Delay Rate (%)')
axes[1].set_title('Major Delay Rate by Day of Week')
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Day of Week Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / "8_day_of_week_analysis.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '8_day_of_week_analysis.png'}")
plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*60)
print("QUESTION 6 ANALYSIS COMPLETE!")
print("="*60)
print(f"\nAll visualizations saved to: {output_dir.absolute()}/")
print("\nKey Findings:")
print(f"  • Total observations: {len(data):,}")
print(f"  • Major delays (≥10 min): {data['major_delay'].sum():,} ({data['major_delay'].sum()/len(data)*100:.2f}%)")
print(f"  • Mean delay: {data['delay_min'].mean():.2f} minutes")
print(f"  • Median delay: {data['delay_min'].median():.2f} minutes")
print(f"  • Class imbalance ratio: {(~data['major_delay']).sum() / data['major_delay'].sum():.1f}:1")
print("\nNext Steps:")
print("  1. Collect more data over time (run scripts regularly)")
print("  2. Merge with GTFS static data for more features")
print("  3. Proceed to Question 7 (baseline model)")
print("  4. Proceed to Question 8 (logistic regression)")
