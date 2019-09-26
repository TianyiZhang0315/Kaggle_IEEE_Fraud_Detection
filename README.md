# Kaggle_IEEE_Fraud_Detection
This Kaggle competition means to find the machine learning solution for fraud detection in transactions.
# Data
Total of 590,000 sample for training, 500,000 for testing (without labels). Training set samples are marked with 'isFraud' binary label. Each training transaction sample contains over 2000 features and some samples have identity information in a separate csv file. 
# Preprocess
## Merge features in Transactions and Identity
Transaction.csv and identity.csv have the same unique key 'TransactionID'. Features are merged into one dataframe by 'TransactionID'.
## Domain knowledge filtering
Remove features that are not relavant to the result of being a fraud transaction, such as 'TransactionID'.
## Fill or remove missing values (can be improved)
Fill all NaN values with 0.
## Remove inconsistent categorical features in train and test (can be improved)
Categorical features might have different amount of unique values in train and test set. My decision is to remove inconsistent features. A better approach might be encode with the union of categorical features in train and test set.
## Encoding for categorical features
Use one-hot endcoding.
## Feature selection with Random Forest feature importance
Feed train set into a random forest model to obtain the feature importance list. A threshold is set to remove features that have low importance. Then standardize and save the data.
## Class imbalance
96% of the samples are negative sample while only 4% are positive. Instead of using the default 0.5 as the threshold of sigmoid function, the ratio of positive samples over negative samples (m+/m-) is used as the new threshold.
# Models
## Logistic Regression
Implement a logistic regression model as baseline model. The optimal parameters are selected by using grid search. 
## Random Forest
Implement a logistic regression model as main model. The grid search parameters are n_estimators, min_samples_split, max_leaf_nodes and min_samples_leaf. 5 fold cross validation is applied.
## SVM with linear kernel
It is nearly impossible to implement SVM with rbf kernel on large data sets because of the time complexcity (O(n^2)). The result of linear kernel SVM has the similar result to logitstic regression.
# Evaluation
## ROC curve (or AUC score)
Use the area under curve score to evaluate the robustness of models. Models iterates on changing of threshold of sigmoid and importance in features selection. 
