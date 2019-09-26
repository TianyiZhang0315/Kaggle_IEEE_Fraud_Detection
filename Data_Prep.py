'''This file contains functions for data preprocessing including feature selection.
This file is also imported by main.py to save trained models.

'''
import pandas as pd
def save_model(model,file_name):
    import pickle
    with open('save/'+file_name+'.pickle', 'wb') as f:
        pickle.dump(model, f)
    print('model saved')
def read_model(file_name):
    import pickle
    with open('save/'+file_name+'.pickle', 'rb') as f:
        model = pickle.load(f)
    return model
    
def one_hot(df):
    dummy_lst = ['R_emaildomain','P_emaildomain','ProductCD','card4','card6','id_12','id_15','id_16','id_23',
                 'id_27','id_28','id_29','id_35','id_36','id_37','id_38','DeviceType']
    dummy_lst += ['M'+str(i) for i in range(1,10)]
    for x in dummy_lst:
        dummies = pd.get_dummies(df[x],prefix = x, dummy_na = True)
        df = df.drop(x,1)
        df = pd.concat([df,dummies],axis = 1)
    return df
def onehot(df):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df)
    return enc
def fill_nan(df):
    return df.fillna(0)
def remove_V_train(df):
    filled1 = df.loc[:,['V'+str(i) for i in range(1,340)]]
    print(filled1.loc[:1])
    c = (filled1 == 0).astype(int).sum(axis=0)#then return(remove from data) label of features that have 70% or more 0 values.
    label_cut = c.loc[c<=filled1.shape[0]*0.1]#get list of features that have more than 10% are NaN(now 0)
    alst = ['V'+str(i) for i in range(1,340)]#get all feature names(list a)
    blst = list(label_cut.index)#get feature names that need to keep(list b) 
    cutlst = [item for item in alst if item not in blst] #substract b from a
    print(len(cutlst))
    df = df.drop(columns = cutlst)#drop features from data
    y_train = df.loc[:,'isFraud']
    x_train = df.drop(columns = ['isFraud'])
    y_train = y_train.to_frame()
    return x_train, y_train,cutlst
def remove_V_test(df,cutlst):
    df = df.drop(columns = cutlst)#drop features from data
    return df
def save_result(predict, file_name):
    outp = test
    outp['isFraud'] = predict
    outp = outp.loc[:,['TransactionID','isFraud']]
    outp.to_csv(file_name+'.csv',index=False)#write into csv
    
#main
if __name__ == '__main__':
    #read in csv
    train = pd.read_csv('D:/ChromeDownload/Kaggle/IEEEfraud/train_transaction.csv')#read in training set
    test = pd.read_csv('D:/ChromeDownload/Kaggle/IEEEfraud/test_transaction.csv')#read in training set
    test_id = pd.read_csv('D:/ChromeDownload/Kaggle/IEEEfraud/test_identity.csv')#read in training set
    train_id = pd.read_csv('D:/ChromeDownload/Kaggle/IEEEfraud/train_identity.csv')#read in training set
    #merge transaction and identity
    train = pd.merge(train, train_id, how='left', on='TransactionID')
    test = pd.merge(test, test_id, how='left', on='TransactionID')
    ##save into csv
    #train.to_csv('combined_train.csv',index=False)#write into csv
    #test.to_csv('combined_test.csv',index=False)#write into csv
    
    #exam number of unique labels in each feature
    unique_label_train = train_id.nunique()
    unique_label_test = test_id.nunique()
    #remove features that are not consistent in number of unique label from features in train set and test set
    train = train.drop(columns = ['DeviceInfo'])
    train = train.drop(columns = ['TransactionID'])
    train = train.drop(columns = ['TransactionDT'])
    train = train.drop(columns = ['id_30'])
    train = train.drop(columns = ['id_31'])
    train = train.drop(columns = ['id_33'])
    train = train.drop(columns = ['id_34'])
    test = test.drop(columns = ['DeviceInfo'])
    test = test.drop(columns = ['TransactionID'])
    test = test.drop(columns = ['TransactionDT'])
    test = test.drop(columns = ['id_30'])
    test = test.drop(columns = ['id_31'])
    test = test.drop(columns = ['id_33'])
    test = test.drop(columns = ['id_34'])
    #one hot encoding
    train = one_hot(train)
    test = one_hot(test)
    #fill NaN
    train = fill_nan(train)
    test = fill_nan(test)
    
    #remove features that contains 90% NaN(omitted when use RF to select features)
#    x_train, y_train,clist = remove_V_train(train)
#    x_test = remove_V_test(test,clist)
#    y_train = train.loc[:,'isFraud']
#    x_train = train.drop(columns = ['isFraud'])
#    y_train = y_train.to_frame()
#    x_train.to_csv('x_train_transaction_f1.csv',index=False)#write into csv
#    y_train.to_csv('y_train_transaction_f.csv',index=False)#write into csv
    
    ###use RF to select features
    y_train = train.loc[:,'isFraud']
    x_train = train.drop(columns = ['isFraud'])
    x_test = test
    from sklearn.ensemble import RandomForestClassifier
    RF = RandomForestClassifier(oob_score=True,n_estimators=40,min_samples_split = 4,verbose = 2)
    RF.fit(x_train,y_train)
    importances = RF.feature_importances_#view feature importance
    cl = list(x_train.columns)
    imp_lst = pd.DataFrame(zip(cl,list(importances)),columns = ['Features','importances'])
    so = sorted(imp_lst.loc[:,'importances'],reverse = True )#transform feature importance into a sorted dataframe
    imp_lst.to_csv('imp_lst.csv',index=False)#write importance list into csv for later use
    imp_lst = pd.read_csv('D:/ChromeDownload/Kaggle/IEEEfraud/imp_lst.csv')#read importance list
    imp_lst1 = imp_lst.sort_values(by=['importances'])
    imp_lst = imp_lst[imp_lst.importances >= 0.01]#set threshold of feature importance
    keep_f = list(imp_lst.loc[:,'Features'])#keep features that have importance score above the threshold
    x_train = x_train.loc[:,keep_f]
    x_test = x_test.loc[:,keep_f]
    x_train.to_csv('x_train_transaction_f.csv',index=False)#write into csv
    x_test.to_csv('x_test_transaction_f.csv',index=False)#write into csv







