# Heart-Attack-Risk-Prediction-Health-Data-Analysis-for-Early-Detection_Machine-Learning

This project focuses on predicting the 10-year risk of coronary heart disease (CHD) based on behavioral, demographic, and medical factors. The goal was to identify at-risk individuals early using a machine learning approach that integrates multiple classification algorithms.

**Tools**: Python, Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn

**Approach**: Evaluated multiple supervised learning models: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, and XGBoost, using accuracy and classification reports. The top three models (Logistic Regression, Random Forest, SVM) were fine-tuned using GridSearchCV, followed by an ensemble stacking method combining all three optimized models for final evaluation.

**Results**: The Random Forest and Logistic Regression models achieved the highest individual accuracy (~85%), while the stacking ensemble slightly improved performance (85.08%). However, results also revealed class imbalance, the model performed well on non-risk cases but struggled to identify positive CHD cases accurately.

**Key Insight**: The ensemble model achieved high accuracy but struggled to detect high-risk patients because the dataset was imbalanced. To improve results, techniques like SMOTE, cost-sensitive learning, or better feature selection could help the model identify heart disease risks more effectively.
