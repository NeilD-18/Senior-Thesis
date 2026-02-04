#!/usr/bin/env python3
"""Visualize Week 1 baseline results."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Read results
results_file = Path('outputs/summary/baseline_results.csv')
df = pd.read_csv(results_file)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Week 1 Baseline Results', fontsize=16, fontweight='bold')

# 1. Test Accuracy Comparison (Classification only)
ax = axes[0, 0]
classification_df = df[df['test_accuracy'].notna()].copy()
classification_df = classification_df.sort_values('test_accuracy', ascending=True)
colors = ['#2ecc71' if acc > 0.8 else '#e74c3c' for acc in classification_df['test_accuracy']]
ax.barh(classification_df['dataset'], classification_df['test_accuracy'], color=colors, alpha=0.7)
ax.set_xlabel('Test Accuracy', fontweight='bold')
ax.set_title('Classification Test Accuracy', fontweight='bold')
ax.set_xlim([0, 1])
ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
for i, (dataset, acc) in enumerate(zip(classification_df['dataset'], classification_df['test_accuracy'])):
    ax.text(acc + 0.02, i, f'{acc:.1%}', va='center', fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# 2. AUROC Comparison
ax = axes[0, 1]
auroc_df = df[df['test_auroc'].notna()].copy()
auroc_df = auroc_df.sort_values('test_auroc', ascending=True)
colors = ['#3498db' if auroc > 0.85 else '#e67e22' for auroc in auroc_df['test_auroc']]
ax.barh(auroc_df['dataset'], auroc_df['test_auroc'], color=colors, alpha=0.7)
ax.set_xlabel('Test AUROC', fontweight='bold')
ax.set_title('Classification Test AUROC', fontweight='bold')
ax.set_xlim([0, 1])
ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, label='0.9 threshold')
for i, (dataset, auroc) in enumerate(zip(auroc_df['dataset'], auroc_df['test_auroc'])):
    ax.text(auroc + 0.02, i, f'{auroc:.3f}', va='center', fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# 3. Val vs Test Accuracy (check overfitting)
ax = axes[1, 0]
classification_df = df[df['test_accuracy'].notna()].copy()
x = range(len(classification_df))
width = 0.35
ax.bar([i - width/2 for i in x], classification_df['val_accuracy'], width, 
       label='Validation', alpha=0.7, color='#3498db')
ax.bar([i + width/2 for i in x], classification_df['test_accuracy'], width, 
       label='Test', alpha=0.7, color='#e74c3c')
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_title('Validation vs Test Accuracy (Overfitting Check)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classification_df['dataset'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0.7, 0.9])

# 4. Regression Results (Airbnb)
ax = axes[1, 1]
regression_df = df[df['test_rmse'].notna()].copy()
if not regression_df.empty:
    metrics = ['RMSE', 'MAE']
    val_values = [regression_df.iloc[0]['val_rmse'], regression_df.iloc[0]['val_mae']]
    test_values = [regression_df.iloc[0]['test_rmse'], regression_df.iloc[0]['test_mae']]
    
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width/2 for i in x], val_values, width, 
           label='Validation', alpha=0.7, color='#9b59b6')
    ax.bar([i + width/2 for i in x], test_values, width, 
           label='Test', alpha=0.7, color='#e67e22')
    ax.set_ylabel('Error (log scale)', fontweight='bold')
    ax.set_title('Airbnb Regression Metrics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, (val, test) in enumerate(zip(val_values, test_values)):
        ax.text(i - width/2, val + 0.01, f'{val:.3f}', ha='center', fontweight='bold')
        ax.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center', fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No regression results', ha='center', va='center', 
            transform=ax.transAxes, fontsize=14)

plt.tight_layout()
plt.savefig('outputs/summary/baseline_plots.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: outputs/summary/baseline_plots.png")
plt.close()

# Create detailed metrics table
print("\n" + "="*80)
print("DETAILED WEEK 1 BASELINE RESULTS")
print("="*80)

print("\n📊 CLASSIFICATION RESULTS:")
print("-" * 80)
classification_df = df[df['test_accuracy'].notna()].copy()
for _, row in classification_df.iterrows():
    print(f"\n{row['dataset'].upper()} ({row['model']})")
    print(f"  Validation: Acc={row['val_accuracy']:.4f}, F1={row['val_f1']:.4f}, AUROC={row['val_auroc']:.4f}")
    print(f"  Test:       Acc={row['test_accuracy']:.4f}, F1={row['test_f1']:.4f}, AUROC={row['test_auroc']:.4f}")
    gap = (row['val_accuracy'] - row['test_accuracy']) * 100
    print(f"  Val-Test Gap: {gap:+.2f}% {'⚠️ Check overfitting' if gap > 2 else '✅ Good generalization'}")

print("\n📈 REGRESSION RESULTS:")
print("-" * 80)
regression_df = df[df['test_rmse'].notna()].copy()
for _, row in regression_df.iterrows():
    print(f"\n{row['dataset'].upper()} ({row['model']})")
    print(f"  Validation: RMSE={row['val_rmse']:.4f}, MAE={row['val_mae']:.4f}")
    print(f"  Test:       RMSE={row['test_rmse']:.4f}, MAE={row['test_mae']:.4f}")
    gap = (row['val_rmse'] - row['test_rmse']) * 100
    print(f"  Val-Test Gap: {gap:+.2f}% {'✅ Good generalization' if abs(gap) < 5 else '⚠️ Check overfitting'}")

print("\n" + "="*80)
print("✅ All Week 1 baseline experiments complete!")
print("="*80)
