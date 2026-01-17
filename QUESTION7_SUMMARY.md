# Question 7: Baseline Model - Summary Report

## Overview

Question 7 requires creating a simple predictor **without machine learning** to establish a performance benchmark. This baseline serves as the minimum performance level that our logistic regression model (Question 8) must exceed.

---

## Dataset

- **Total records:** 46,344 observations
- **Training set:** 37,075 records (80%)
- **Test set:** 9,269 records (20%)
- **Test set delay rate:** 2.74% (254 major delays out of 9,269 observations)

---

## Three Baseline Strategies Tested

### 1. Majority Class Baseline

**Strategy:** Always predict "No Major Delay"

**Rationale:** Since 96.67% of trips have no major delay, simply predicting the majority class should give high accuracy.

**Results:**

- ✅ Accuracy: **97.26%** (very high!)
- ❌ Precision: **0.00** (undefined - no positive predictions)
- ❌ Recall: **0.00%** (never identifies actual delays)
- ❌ F1 Score: **0.000**

**Confusion Matrix:**

```
                  Predicted
                  No Delay    Delay
Actual  No Delay    9,015       0
        Delay         254       0
```

**Analysis:**

- This baseline demonstrates a critical problem with imbalanced datasets
- While accuracy is very high (97.26%), the model is **completely useless**
- It has **0% recall** - it never identifies any actual delays
- This shows why **accuracy alone is misleading** for imbalanced classification

---

### 2. Route-Based Baseline ⭐ **BEST PERFORMER**

**Strategy:** Predict delay if the route has a historical delay rate > 5%

**Rationale:** Some routes may be more prone to delays due to traffic patterns, route length, or other factors.

**Implementation:**

1. Calculate historical delay rate for each route in training data
2. Identify 30 routes with delay rate > 5%
3. Predict "Major Delay" for any trip on these high-delay routes

**Results:**

- Accuracy: **83.83%**
- Precision: **13.23%** (low - many false positives)
- ✅ Recall: **88.19%** (high - catches most actual delays!)
- ✅ F1 Score: **0.2301** (best among baselines)

**Confusion Matrix:**

```
                  Predicted
                  No Delay    Delay
Actual  No Delay    7,546     1,469
        Delay          30       224
```

**Analysis:**

- **Best baseline** with F1 score of 0.2301
- Excellent recall (88.19%) - catches 224 out of 254 actual delays
- Low precision (13.23%) - many false alarms (1,469 false positives)
- Trade-off: Lower accuracy but much more useful for identifying delays
- Shows that **route information has predictive value**

---

### 3. Time-Based Baseline (Rush Hour)

**Strategy:** Predict delay during rush hours (7-9 AM, 4-6 PM)

**Rationale:** Delays might be more common during peak commute times.

**Results:**

- Accuracy: **97.26%**
- Precision: **0.00**
- Recall: **0.00%**
- F1 Score: **0.000**

**Confusion Matrix:**

```
                  Predicted
                  No Delay    Delay
Actual  No Delay    9,015       0
        Delay         254       0
```

**Analysis:**

- Performed identically to majority class baseline
- The data collected didn't include rush hour periods (or rush hour doesn't predict delays well)
- Shows that **time alone is not a strong predictor** in this dataset
- Note: This may be due to limited data collection period

---

## Comparison Summary

| Baseline | Accuracy | Precision | Recall | F1 Score |
|----------|----------|-----------|--------|----------|
| **Majority Class** | 97.26% | 0.000 | 0.00% | 0.000 |
| **Route-Based** ⭐ | 83.83% | 0.132 | 88.19% | **0.2301** |
| **Time-Based** | 97.26% | 0.000 | 0.00% | 0.000 |

---

## Key Findings

### 1. Accuracy is Misleading

- The majority class baseline achieves 97.26% accuracy but is useless
- **Lesson:** For imbalanced datasets, focus on Precision, Recall, and F1 Score

### 2. Route Information Matters

- The route-based baseline achieves F1 = 0.2301
- This shows that **route_id is a valuable feature** for prediction
- Some routes are consistently more prone to delays

### 3. Recall vs Precision Trade-off

- Route-based baseline: High recall (88%), low precision (13%)
- This means: Catches most delays but has many false alarms
- For commuters, this might be acceptable (better to be warned unnecessarily than miss a delay)

### 4. Class Imbalance Challenge

- Only 2.74% of test observations are major delays
- This makes achieving good precision very difficult
- Any model predicting delays frequently will have low precision

---

## Benchmark for Question 8

**The logistic regression model (Question 8) must beat:**

- **F1 Score: 0.2301**
- **Recall: 88.19%** (ideally maintain or improve)
- **Precision: 13.23%** (should improve this)

**Success criteria for Question 8:**

- ✅ F1 Score > 0.23
- ✅ Recall ≥ 85% (maintain ability to catch delays)
- ✅ Precision > 15% (reduce false alarms)

---

## Implications for Question 8 (Logistic Regression)

### Features to Include

1. **route_id** - Proven to be predictive (route-based baseline worked)
2. **hour, day_of_week** - Temporal features (even if rush hour alone didn't work)
3. **stop_id** - Some stops may be more delay-prone
4. **Combined features** - Route + time interactions

### Strategies to Try

1. **Class weighting:** Use `class_weight='balanced'` to handle imbalance
2. **Feature engineering:** Create interaction features (route × hour)
3. **Threshold tuning:** Adjust decision threshold to balance precision/recall
4. **SMOTE:** Consider oversampling minority class

### Expected Improvement

- Logistic regression should combine route + time + stop information
- Should achieve better precision while maintaining high recall
- Target: F1 Score > 0.30 (30% improvement over baseline)

---

## Visualizations Created

1. **Baseline Comparison Chart** (`9_baseline_comparison.png`)
   - Shows all metrics side-by-side
   - Clearly demonstrates route-based baseline superiority

2. **Confusion Matrices** (`10_baseline_confusion_matrices.png`)
   - Visual comparison of prediction patterns
   - Shows route-based baseline catches most delays (224/254)

---

## For Your Report

### What to Write

**Section: Baseline Model (Question 7)**

"We implemented three baseline predictors without machine learning to establish a performance benchmark:

1. **Majority Class Baseline:** Always predicts no delay, achieving 97.26% accuracy but 0% recall, demonstrating that accuracy alone is misleading for imbalanced datasets.

2. **Route-Based Baseline:** Predicts delays for routes with historical delay rates above 5%. This achieved the best performance with an F1 score of 0.2301 and recall of 88.19%, though precision was low at 13.23%.

3. **Time-Based Baseline:** Predicts delays during rush hours, but performed identically to the majority class baseline, suggesting time alone is not a strong predictor in our dataset.

The route-based baseline establishes our benchmark: **F1 = 0.2301**. This demonstrates that route information has predictive value, and our logistic regression model (Question 8) must exceed this performance to be considered successful."

### Key Points to Emphasize

- Why accuracy is misleading (majority class achieves 97% but is useless)
- Route information is predictive (F1 = 0.23)
- Benchmark established for evaluating ML models
- Trade-off between precision and recall

---

## Files Generated

- `question7_baseline_model.py` - Complete implementation
- `results/baseline_comparison.csv` - Numerical results
- `visualizations/9_baseline_comparison.png` - Metrics comparison
- `visualizations/10_baseline_confusion_matrices.png` - Confusion matrices

---

## Next Steps

✅ **Question 7 Complete!**

🔜 **Ready for Question 8:** Logistic Regression

- Goal: Beat F1 score of 0.2301
- Use route, time, and stop features
- Address class imbalance
- Tune hyperparameters

---

## Summary

**Best Baseline:** Route-Based Predictor

- **F1 Score:** 0.2301
- **Recall:** 88.19% (catches most delays)
- **Precision:** 13.23% (many false alarms)

**This is the benchmark to beat in Question 8!** 🎯
