import pandas as pd
import Data_Prep as dp
def standardize(x_train,x_test):#Standardization(use train set parameters to fit test set)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train,x_test


def logisticRegression(x_train,y_train,x_test,param_dict = {},threshold = 0.5):#logistic regression model
    from sklearn.linear_model import LogisticRegression 
    #param_dict = {'max_iter':300}
    print('Build model')
    model = LogisticRegression(**param_dict)
    print('Start training')
    model.fit(x_train,y_train)
    print('training finished')
    print('start prediction')
    LR_predict = model.predict_proba(x_test)
    print('prediction finished')
    #threshold = 0.0363 #set decision threshold for class imbalance (threshold = positive sample/negative sample)
    LR_predicted = (LR_predict [:,1] >= threshold).astype('int')
    dp.save_result(LR_predicted, 'LR_result13_threshold00363')#save prediction

def RandomForestGS(x_train,y_train,x_test,param_dict = {}, threshold = 0.5):#Random Forest (gridSearch)

    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    #param_dict = {'n_estimators':[240]}
    print('Build model')
    RF = RandomForestClassifier(oob_score=True,n_estimators=240,verbose = 2,min_samples_split=10,max_leaf_nodes = 1400,
                            min_samples_leaf = 30,n_jobs = 8)#weight = 25
    clf = GridSearchCV(estimator = RF, 
                       param_grid = param_dict, scoring='roc_auc',cv = 5)
    print('Start training')
    clf.fit(x_train, y_train)
    print('training finished')
    best_score2 = clf.best_score_
    print('Cross Validation Result:',clf.cv_results_,'\nBest Parameter:', clf.best_params_)
    
def RandomForest(x_train,y_train,x_test, threshold = 0.5):#Random Forest (without gridSearch)

    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    #param_dict = {'n_estimators':[240]}
    print('Build model')
    RF = RandomForestClassifier(oob_score=True,n_estimators=240,verbose = 2,min_samples_split=10,max_leaf_nodes = 1400,
                            min_samples_leaf = 30,n_jobs = 8)#weight = 25
    
    print('Start training')
    RF.fit(x_train, y_train)
    print('training finished')
    print('start prediction')
    RF_predict = RF.predict_proba(x_test)
    print('prediction finished')
    #threshold = 0.0363 #set decision threshold for class imbalance (threshold = positive sample/negative sample)
    RF_predicted = (RF_predict [:,1] >= threshold).astype('int')
    dp.save_result(RF_predicted, 'RF_result13_threshold00363')#save prediction

def svm(x_train,y_train,x_test, threshold = 0.5):#SVM with linear kernel (similiar to LogisticRegression)
    from sklearn import svm
    svm = svm.SVC( kernel = 'linear')
    print('Start training')
    svm.fit(x_train,y_train)
    print('training finished')
    print('start prediction')
    svm_predict = svm.predict_proba(x_test)
    print('prediction finished')
    #threshold = 0.0363 #set decision threshold for class imbalance (threshold = positive sample/negative sample)
    svm_predicted = (svm_predict [:,1] >= threshold).astype('int')
    dp.save_result(svm_predicted, 'SVM_result13_threshold00363')#save prediction

def xgb(x_train,y_train,x_test, threshold = 0.5):#XGBoost model
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from xgboost import plot_importance
    model = XGBClassifier(n_estimators=300)
    print('Start training')
    model.fit(x_train,
          y_train)
    print('training finished')
    print('start prediction')
    xgb_predict = model.predict_proba(x_test)
    print('prediction finished')
    #threshold = 0.0363
    xgb_predicted = (xgb_predict [:,1] >= threshold).astype('int')
    dp.save_result(xgb_predicted, 'xgb_result237_threshold00363')
    
    
#__main__
if __name__ = '__main__':
    x_train = pd.read_csv('D:/ChromeDownload/Kaggle/IEEEfraud/x_train_transaction_f2.csv')#read in training set
    y_train = pd.read_csv('D:/ChromeDownload/Kaggle/IEEEfraud/y_train_transaction_f.csv')
    x_test = pd.read_csv('D:/ChromeDownload/Kaggle/IEEEfraud/x_test_transaction_f2.csv')
    #from sklearn.model_selection import train_test_split #train test split
    #X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    param_dict = {'n_estimators':[240]}
    RandomForestGS(x_train,y_train,x_test,param_dict,0.0363)