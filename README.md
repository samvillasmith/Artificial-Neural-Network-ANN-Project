# Customer Churn Prediction - ANN Model

A deep learning application for predicting bank customer churn using an Artificial Neural Network with class weighting to handle imbalanced data.

## 📊 Model Performance

**Final Metrics (Test Set):**
- **Accuracy:** 81.5%
- **Recall:** 67.6% (catches 68% of churners)
- **Precision:** 53.6% (54% of churn predictions are correct)
- **AUC-ROC:** 0.85 (excellent discrimination)

**Confusion Matrix:**
- True Negatives: 1,355 (correctly predicted non-churners)
- True Positives: 275 (correctly predicted churners)
- False Positives: 238 (false alarms)
- False Negatives: 132 (missed churners)

## 🏗️ Architecture

```
Input Layer (12 features)
    ↓
Dense(128, relu) + Dropout(0.3)
    ↓
Dense(64, relu) + Dropout(0.3)
    ↓
Dense(32, relu) + Dropout(0.2)
    ↓
Output Layer (1, sigmoid)
```

**Total Parameters:** 6,785

## 🎓 Lessons Learned

### 1. Class Imbalance is a Silent Killer

**The Problem:**
- Dataset: 79.63% non-churners, 20.37% churners
- Initial model without class weights: ~6% recall
- The model learned to just predict "no churn" for everyone

**The Solution:**
- Implemented class weights (balanced)
- Penalized the model more heavily for missing churners
- Recall improved from 6% → 67.6%

**Key Takeaway:** 
> Accuracy is a terrible metric for imbalanced datasets. A model that predicts "no churn" for everyone gets 80% accuracy but is completely useless. Always check precision, recall, and confusion matrix.

### 2. More Data ≠ Better Data

**What We Discovered:**
- Even with class weights and good architecture, some "obviously risky" customers only scored 35% churn probability
- Example: Credit score 400, balance $120, tenure 0, inactive → 34.7% churn risk

**Why This Happens:**
- The dataset only has 12 features
- Missing critical information:
  - Customer satisfaction scores
  - Customer service interactions/complaints
  - Recent life events (marriage, job loss, relocation)
  - Competitive offers received
  - Digital engagement metrics
  - Transaction patterns
  - Recent fee charges

**Key Takeaway:**
> A model can only be as good as the data it's trained on. If "risky-looking" customers in the training data actually didn't churn (perhaps they had other factors keeping them), the model learns this pattern.

### 3. The Curse of Binary Classification

**The Challenge:**
- Real-world churn is a spectrum, not binary
- Customers at 34% risk are different from those at 67% risk
- But they're both just "predicted no churn" at a 0.5 threshold

**Our Solution:**
- Made threshold adjustable (default: 0.25)
- Added risk tiers: Critical (>70%), High (>50%), Elevated (>threshold)
- Show raw probability, not just binary prediction

**Key Takeaway:**
> Always show probability scores, not just yes/no predictions. Business stakeholders need to know the difference between "barely below threshold" and "definitely safe."

### 4. Early Stopping Can Be Tricky

**What Happened:**
- Model saved at epoch 18 (AUC: 0.866) was excellent
- Model saved at epoch 20 after early stopping was garbage
- Validation AUC slightly declined, triggering early stop

**The Issue:**
- Early stopping restored weights from epoch 5 (patience=15, looking back)
- This wasn't actually the best model - epoch 18 was

**Our Fix:**
- Use `ModelCheckpoint` to save best model separately
- Load `best_churn_model.h5`, not the final model
- Consider using multiple metrics for early stopping

**Key Takeaway:**
> ModelCheckpoint with `save_best_only=True` is more reliable than early stopping's `restore_best_weights`. Save the best model as you go.

### 5. Feature Engineering Matters More Than Architecture

**What We Tried:**
- Initial architecture: 64→32→1 (2,945 params)
- Improved architecture: 128→64→32→1 (6,785 params)
- Performance gain: Minimal (maybe 2-3% better)

**What Actually Worked:**
- Class weights: 6% recall → 67% recall (1000% improvement!)
- Proper preprocessing and feature scaling
- Correct column ordering

**Key Takeaway:**
> Going from 2 layers to 4 layers gave marginal gains. Fixing class imbalance gave 10x gains. Focus on data quality and class balance before obsessing over architecture.

## 🚫 Dataset Limitations

### 1. **Severe Class Imbalance**
- **Issue:** 80/20 split (non-churners/churners)
- **Impact:** Model bias toward majority class
- **Mitigation:** Class weights, but still impacts learning
- **Better Solution:** Collect more churner data or use SMOTE

### 2. **Limited Feature Set**

**What We Have:**
- Demographics: Geography, Gender, Age
- Financial: Credit Score, Balance, Salary
- Banking: Tenure, Products, Credit Card, Active Status

**What We're Missing:**
| Missing Feature | Why It Matters | Expected Impact |
|----------------|----------------|-----------------|
| Customer Satisfaction | Unhappy customers churn | High |
| Complaint History | Complaints predict churn | High |
| Customer Service Contacts | Problems indicate risk | High |
| Competitive Offers | External triggers | High |
| Transaction Frequency | Engagement indicator | Medium |
| Recent Fee Charges | Dissatisfaction source | Medium |
| Digital Banking Usage | Modern engagement | Medium |
| Life Events (marriage, move) | Major life changes | Medium |
| Relationship Length | Loyalty indicator | Low |
| Cross-selling History | Engagement depth | Low |

**Real Example:**
A customer with credit score 400 and $120 balance shows only 35% churn risk. In reality, this should be much higher. Why isn't it?
- Maybe they're deeply engaged digitally
- Maybe they have a great relationship with their banker
- Maybe they have no better options
- **Or maybe:** The dataset just doesn't capture the full picture

### 3. **Temporal Limitations**

**Issues:**
- No time-series data
- Can't capture trends (declining balance, increasing complaints)
- Single snapshot in time
- No seasonality information

**Example:**
- Customer A: Balance $50k (up from $30k last year) → Low risk
- Customer B: Balance $50k (down from $100k last year) → High risk
- **Our model sees them as identical**

### 4. **No Causality, Only Correlation**

**The Problem:**
- Model learns: "Low balance → churn"
- Reality: "Planning to leave → withdraw funds → churn"
- Low balance is a symptom, not a cause

**Implications:**
- Can't prevent churn by artificially increasing balance
- Need to address root causes (dissatisfaction, better offers)
- Model tells us WHO will churn, not WHY or HOW to prevent it

### 5. **Geographic Bias**

**Distribution in Dataset:**
- France: 50.14%
- Germany: 25.09%
- Spain: 24.77%

**Issues:**
- May not generalize to other regions
- Cultural differences in banking behavior not captured
- Economic conditions vary by country
- Banking regulations differ

### 6. **Static Demographics**

**Problem:**
- Age, gender, geography rarely change
- These are weak predictors alone
- Need behavioral features for better predictions

**Better Alternatives:**
- Change in behavior (spending patterns)
- Engagement metrics (logins, transactions)
- Service usage (ATM, mobile, branch)

### 7. **Survivor Bias**

**Potential Issue:**
- Dataset only includes current customers
- Doesn't include customers who already churned
- May underrepresent certain churn patterns
- Historical churn data would be more valuable

### 8. **No External Factors**

**Missing Context:**
- Economic conditions (recession, boom)
- Competitor actions (better rates offered)
- Regulatory changes (new banking laws)
- Market events (financial crisis)

## 🎯 Recommendations for Production

### If Using This Model:

1. **Set threshold based on business needs:**
   - **Low threshold (0.25):** Catch more churners, more false alarms (broad retention campaign)
   - **Medium threshold (0.35):** Balanced approach (targeted outreach)
   - **High threshold (0.50):** Only most confident predictions (high-touch retention)

2. **Combine with business rules:**
   ```python
   if prediction_prob > 0.35:
       if balance < 1000 and tenure < 1:
           priority = "CRITICAL"
       elif num_complaints > 2:  # If we had this data!
           priority = "HIGH"
   ```

3. **Monitor model performance:**
   - Track actual churn vs predictions monthly
   - Retrain if recall drops below 60%
   - Update class weights as churn rate changes

4. **Use in conjunction with:**
   - Customer satisfaction surveys
   - Support ticket analysis
   - Manual risk assessment by account managers

### If Building a Better Model:

1. **Collect More Features:**
   - Customer service interactions
   - Transaction patterns over time
   - Digital engagement metrics
   - Satisfaction scores
   - Complaint history

2. **Use Time-Series Data:**
   - Last 6-12 months of data per customer
   - LSTM or Transformer architecture
   - Capture trends and seasonality

3. **Try Other Approaches:**
   - **XGBoost/LightGBM:** Often outperform neural networks on tabular data
   - **Ensemble Methods:** Combine multiple models
   - **SMOTE:** Synthetic oversampling for minority class

4. **A/B Test in Production:**
   - Start with a small cohort
   - Measure intervention success rate
   - Iterate based on real results

## 🔧 Technical Debt & Future Work

### Quick Wins:
- [ ] Add SHAP values for explainability
- [ ] Implement model versioning
- [ ] Add data validation pipeline
- [ ] Create automated retraining pipeline
- [ ] Add model monitoring dashboard

### Long-term Improvements:
- [ ] Collect time-series features
- [ ] Build feedback loop (track intervention results)
- [ ] Experiment with XGBoost/LightGBM
- [ ] Implement SMOTE or other sampling techniques
- [ ] Add feature importance analysis
- [ ] Create customer segmentation model first

## 📈 Success Metrics in Production

Don't just track model metrics - track business outcomes:

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| Retention Rate (Predicted Churners) | >40% | Are interventions working? |
| False Alarm Cost | <$50/customer | Are we wasting resources? |
| Revenue Saved | >$500k/year | Is this worth the effort? |
| Model Recall | >65% | Are we catching most churners? |
| Intervention Success Rate | >30% | Can we actually save them? |

## 🎓 Final Thoughts

**What Worked:**
- ✅ Class weights solved the imbalance problem
- ✅ Dropout prevented overfitting
- ✅ Adjustable threshold enables business flexibility
- ✅ Model achieves 67.6% recall (captures most churners)

**What Didn't:**
- ❌ Can't predict churn perfectly without better features
- ❌ Some "obvious" high-risk customers score low
- ❌ Architectural improvements gave minimal gains
- ❌ Dataset limitations can't be overcome by better modeling

**The Honest Truth:**
This model is **good enough for a proof-of-concept** but needs significant improvements for production use:
- 67.6% recall means we're catching 2 out of 3 churners ✅
- 53.6% precision means about half our alerts are false alarms ⚠️
- Missing critical features limits prediction quality ❌
- Need feedback loop to measure intervention success ❌

**The Bottom Line:**
> The best ML model can't overcome bad data. Before spending weeks tuning hyperparameters, spend days collecting better features. A simple model with great features beats a complex model with poor features every time.

## 📚 Resources

- [Dealing with Imbalanced Data](https://imbalanced-learn.org/)
- [Understanding Class Weights](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
- [When to Use Neural Networks vs. Gradient Boosting](https://towardsdatascience.com/)
- [Churn Prediction Best Practices](https://www.kaggle.com/competitions/churn-prediction)


Built as a practice project to understand class imbalance, neural network architecture, and the gap between model metrics and business value.

