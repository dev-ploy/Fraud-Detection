# Fraud Detection Project:

## 1. Data Cleaning
- **Missing Values:**
  ```python
  df.isnull().sum()
  df.dropna(subset='isFraud', inplace=True)
  ```
- **Outliers:**
  ```python
  sns.boxplot(data=df[df['amount']<50000], x="isFraud", y="amount")
  ```
- **Multi-collinearity:**
  ```python
  corr = df[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud']].corr()
  sns.heatmap(corr, annot=True)
  ```

## 2. Model Description
- **Models Used:** Logistic Regression, SVM, Random Forest
- **Pipeline Example:**
  ```python
  preprocessor = ColumnTransformer([
      ("num", StandardScaler(), numerical),
      ("cat", OneHotEncoder(drop="first", handle_unknown='ignore'), categorical)
  ])
  pipeline = Pipeline([
      ("prep", preprocessor),
      ("clf", RandomForestClassifier(class_weight="balanced"))
  ])
  ```

## 3. Variable Selection
- **Selected Features:**
  ```python
  categorical = ['type']
  numerical = ['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','balanceDiffOrig','balanceDiffDest']
  X = df_model.drop('isFraud', axis=1)
  y = df_model['isFraud']
  ```

## 4. Model Performance
- **Evaluation:**
  ```python
  print(classification_report(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred))
  pipeline.score(X_test, y_test)*100
  ```
- **Scores:**
  - Logistic Regression: 83%
  - SVM: 95%
  - Random Forest: 99%

## 5. Key Predictive Factors
- **Transaction Type, Amount, Balance Differences**
  ```python
  fraud_by_type = df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
  ```

## 6. Factor Reasoning
- These factors make sense because fraud often involves abnormal transaction types and suspicious balance changes.

## 7. Prevention Recommendations
- Use real-time monitoring, anomaly detection, and regular model retraining.

## 8. Effectiveness Measurement
- Track fraud rate reduction, false positive/negative rates, and model accuracy after implementation.

---

## Why SVM/Random Forest are More Efficient than Naive Bayes
- **Naive Bayes** assumes feature independence, which is rarely true in financial data (e.g., balances and amounts are correlated).
- **SVM** can model complex, non-linear boundaries and is robust to outliers.
- **Random Forest** handles feature interactions, non-linearity, and is less sensitive to noise and outliers.
- Both SVM and Random Forest showed much higher accuracy (95% and 99%) compared to typical Naive Bayes results in fraud detection tasks.

---

## References to Code
- See `notebooks/EDA_and_Model_Building.ipynb` for full code and analysis.
- See `webapp/app.py` for deployment example.
