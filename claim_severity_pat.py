import csv
import pandas as pd
import time
import numpy as np
import sklearn.cross_validation
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
import os
os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'

def f_regression(X,Y):
   import sklearn
   return sklearn.feature_selection.f_regression(X,Y,center=False)

def main():
    start_time = time.time()
    try:
        df2 = pd.read_csv('train.csv', index_col=0)
    except:
        print ("AN ERROR OCCURED, MAKE SURE THE train.csv FILE IS AVAILABLE WHERE(IN THE DIRECTORY) THE CODE IS EXECUTED.")
    else:
        print ("")
        print ("train.csv FILE SUCCESSFULLY READ")
        print ("")
    split = 116
    print ("Would you like to use:-")
    print ("1) Regular Transformation(FASTER)(RECOMMENDED)")
    print ("2) One Hot Encoding")
    ch = int(input ("Enter your choice:- (1 or 2)\n"))
    
    #####################################################
    
    if ch == 1:
        print ("")
        print ("Regular method for data transformaion(PROCESSING....)")
        print ("")
        data = df2.iloc[:,:split]
        columns = data.columns  
        pl = {}
        num = 0
        index = data.index
        for i in columns:
            da = list(data[i])
            for j in range(0,len(da)):
                if da[j] not in pl:
                    num = num + 1
                    pl[da[j]] = [num,1]
                    da[j] = num
                if da[j] in pl:
                    pl[da[j]][1] = pl[da[j]][1] + 1
                    da[j] = pl[da[j]][0]
            df2[i] = da
        
        train1, test1 = sklearn.cross_validation.train_test_split(df2, train_size = 0.8)
        learn_final(train1, test1, 0, 130)
        del train1, test1
        
    #####################################################################################
    
    elif ch == 2:
        print ("")
        print ("One Hot Encoding method for data transformaion(PROCESSING....)")
        print ("")
        index = df2.index
        cols = df2.columns
    
        lab = []
        for i in range(0,split):
            training = df2[cols[i]].unique()
            lab.append(list(set(training)))   
    
        cats = []
        for i in range(0, split):
            lab_en = LabelEncoder()
            lab_en.fit(lab[i])
            feat = lab_en.transform(df2.iloc[:,i])
            feat = feat.reshape(df2.shape[0], 1)
            onehot_en = OneHotEncoder(sparse=False,n_values=len(lab[i]))
            feat = onehot_en.fit_transform(feat)
            cats.append(feat)
        encoded_cats = np.column_stack(cats)
        dataset_encoded = np.concatenate((encoded_cats,df2.iloc[:,116:130].values),axis=1)
        del cats
        del feat
        del encoded_cats
    
        Y = df2["loss"]
        del df2
    
        r, c = dataset_encoded.shape
        df2_new = pd.DataFrame(data=dataset_encoded, index=index, columns=[x for x in range(0,c)])
        del dataset_encoded
    
        df2_new['loss'] = Y
        del Y
    
        X = df2_new.iloc[:,:c]
        Y = df2_new['loss']
        del df2_new
    
        X_new = SelectKBest(score_func=f_regression,k=800).fit_transform(X, Y)
        del X
    
        X_scaled = preprocessing.scale(X_new)
        del X_new
    
        df_new = pd.DataFrame(data=X_scaled, index=index, columns=[x for x in range(0,800)])
        df_new["loss"] = Y
        del Y
    
        train2, test2 = sklearn.cross_validation.train_test_split(df_new, train_size = 0.8)
    
        #For predicting the given test dataset
        ###learn2(train2, test2, 0, 800, df_new) 
        del df_new
    
        learn_final(train2, test2, 0, 800)
        del train2, test2
    else:
        print ("YOU HAVE ENTERED THE WORNG CHOICE, PLEASE RE-RUN THE CODE AGAIN")
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
def learn_final(train, test, n,m):
    print ("")
    print ("Predicting test subset out of the training set:-")
    print ("")
    print ("Gradient Boosting Regressor(PROCESSING....):-")
    print ("")
    gradientboosting(train,test, 47, n,m)
    
# For 10-fold cross validation    
def call_regressor(dfList, n, m):
    print ("10-fold Cross Validation Results")
    lasso_mean = []
    ridge_mean = []
    linear_mean = []
    elasticnet_mean = []
    extratree_mean = []
    gradientboosting_mean = []
    #randomforest_mean = []
    #bayesianridge_mean = []
    #xgboost_mean = []
        
    for i in range(0, len(dfList)):
        test = dfList[i]
        train_arr = []
        for j in range(0, len(dfList)):
            if i!=j:
                train_arr.append(dfList[j])
        train = pd.concat(train_arr)
        
        lasso_mean.append(lasso(train, test, 0, n, m))
        ridge_mean.append(ridge(train, test, 0, n, m))
        linear_mean.append(linear(train, test, 0, n, m))
        elasticnet_mean.append(elasticnet(train, test, 0, n, m))
        extratree_mean.append(extratree(train, test, 0, n, m))
        gradientboosting_mean.append(gradientboosting(train,test, 0, n, m))
        #randomforest_mean.append(randomforest(train,test, 0, n))
        #bayesianridge_mean.append(bayesianridge(train,test, 0, n, m))
        ###xgboost_mean.append(xgboost(train,test, 0, n, m))
    
    print ("LASSO PREDICTOR")
    print (np.mean(lasso_mean))
    
    print ("RIDGE PREDICTOR")
    print (np.mean(ridge_mean))
    
    print ("LINEAR PREDICTOR")
    print (np.mean(linear_mean))
    
    print ("ELASTICNET PREDICTOR")
    print (np.mean(elasticnet_mean))
    
    print ("Extra Tree Bagging PREDICTOR")
    print (np.mean(extratree_mean))
    
    print ("SGD Rrgressor Boosting")
    print (np.mean(gradientboosting_mean))
    
    #print ("Random Forest Regression")
    #print (np.mean(randomforest_mean))
        
    #print ("Bayesian Ridge Regression")
    #print (np.mean(bayesianridge_mean))
    
    #print ("xg boost Regression")
    #print (np.mean(xgboost_mean))
    

# Learning using the best predictor
def learn2(train, test, n,m, df):
    result1 = 0
    result2 = 0
    ###result3 = 0
    result4 = 0
    result5 = 0
    result6 = 0
    ###result7 = 0
    ###result8 = 0
    ###result9 = 0
    r3=0
    
    print ("LASSO PREDICTOR")
    result1 = lasso(train, test, 0, n,m)
    print ("RESULT MEAN ABSOLUTE ERROR = ",result1)
    ##r3 = lasso(train, test, 2, n,m)
    ##print ("RESULT MEAN ABSOLUTE ERROR WITH RFE= ",r3)
    print ("____")
    
    print ("RIDGE PREDICTOR")
    result2 = ridge(train, test, 0, n,m)
    print ("RESULT MEAN ABSOLUTE ERROR = ",result2)
    ##r3 = ridge(train, test, 2, n,m)
    ##print ("RESULT MEAN ABSOLUTE ERROR WITH RFE= ",r3)
    print ("____")
    
    ##print ("LINEAR PREDICTOR")
    ##result3 = linear(train, test, 0, n,m)
    ##print ("RESULT MEAN ABSOLUTE ERROR = ",result3)
    ##r3 = linear(train, test, 2, n,m)
    ##print ("RESULT MEAN ABSOLUTE ERROR WITH RFE= ",r3)
    ##print ("____")
    
    print ("ELASTICNET PREDICTOR")
    result4 = elasticnet(train, test, 0, n,m)
    print ("RESULT MEAN ABSOLUTE ERROR = ",result4)
    ##r3 = elasticnet(train, test, 2, n,m)
    ##print ("RESULT MEAN ABSOLUTE ERROR WITH RFE= ",r3)
    print ("____")
    
    print ("Extra Tree Bagging PREDICTOR")
    result5 = extratree(train, test, 0, n,m)
    print ("RESULT MEAN ABSOLUTE ERROR = ",result5)
    ##r3 = extratree(train, test, 2, n,m)
    ##print ("RESULT MEAN ABSOLUTE ERROR WITH RFE= ",r3)
    print ("____")
    
    print ("Gradient Boosting Regressor")
    result6 = gradientboosting(train,test, 0, n,m)
    print ("RESULT MEAN ABSOLUTE ERROR = ",result6)
    ##r3 = gradientboosting(train, test, 2, n,m)
    ##print ("RESULT MEAN ABSOLUTE ERROR WITH RFE= ",r3)
    print ("____")
    
    ##print ("Random Forest Regressor")
    ##print ("NO RF")
    ##result7 = randomforest(train,test, 0, n,m)
    ##print ("RESULT MEAN ABSOLUTE ERROR = ",result7)
    ##r3 = randomforest(train, test, 2, n,m)
    ##print ("RESULT MEAN ABSOLUTE ERROR WITH RFE= ",r3)
    ##print ("____")
    
    ##print ("Bayesian Ridge Regression")
    ##result8 = bayesianridge(train,test, 0, n,m)
    ##print ("RESULT MEAN ABSOLUTE ERROR = ",result8)
    ##r3 = bayesianridge(train, test, 2, n,m)
    ##print ("RESULT MEAN ABSOLUTE ERROR WITH RFE= ",r3)
    ##print ("____")
    
    ###print ("xg boost Regression")
    ###result9 = xgboost(train,test, 0, n,m)
    ###print ("RESULT MEAN ABSOLUTE ERROR = ",result9)
    ###print ("____")
    
    mini = min([result1,result2,result4,result5,result6])
    
    mini = min([result6])
    print ("WE ARE GOING TO USE THE BEST REGRESSOR WHICH IS - ", end='')
    if mini == result1:
        print ("LASSO REGRESSION")
        testlearn(1,df,m)
    elif mini == result2:
        print ("RIDGE REGRESSION")
        testlearn(2,df,m)
    elif mini == result3:
        print ("LINEAR REGRESSION")
        testlearn(3,df,m)
    elif mini == result4:
        print ("ELASTICNET REGRESSION")
        testlearn(4,df,m)
    elif mini == result5:
        print ("EXTRA TREE REGRESSION")
        testlearn(5,df,m)
    elif mini == result6:
        print ("GRADIENT BOOSTING REGRESSION")
        testlearn(6,df,m)
    elif mini == result7:
        print ("RANDOM FOREST REGRESSION")
        testlearn(7,df,m)
    elif mini == result8:
        print ("BAYESIAN RIDGE REGRESSION")
        testlearn(8,df,m)
    elif mini == result9:
        print ("XGBOOST REGRESSION")
        testlearn(9,df,m)
    
def testlearn(num,df,m):
    df2 = pd.read_csv('test.csv', index_col=0)
    split = 116
    print ("method for test data transformaion")
    index = df2.index
    cols = df2.columns
    
    lab = []
    for i in range(0,split):
        training = df2[cols[i]].unique()
        lab.append(list(set(training)))   
    
    cats = []
    for i in range(0, split):
        lab_en = LabelEncoder()
        lab_en.fit(lab[i])
        feat = lab_en.transform(df2.iloc[:,i])
        feat = feat.reshape(df2.shape[0], 1)
        onehot_en = OneHotEncoder(sparse=False,n_values=len(lab[i]))
        feat = onehot_en.fit_transform(feat)
        cats.append(feat)
    
    encoded_cats = np.column_stack(cats)
    dataset_encoded = np.concatenate((encoded_cats,df2.iloc[:,116:130].values),axis=1)
    del cats
    del feat
    del encoded_cats
    del df2
    
    r, c = dataset_encoded.shape
    df2_new = pd.DataFrame(data=dataset_encoded, index=index, columns=[x for x in range(0,c)])
    del dataset_encoded
    
    X = df2_new.iloc[:,:c]
    del df2_new
    print ("FOR 800 features and scaled vectors - FOR TESTING")
    X_new = X
    del X
    X_scaled = preprocessing.scale(X_new)
    del X_new
    
    try:
        df_new = pd.DataFrame(data=X_scaled, index=index, columns=[x for x in range(0,c)])
    except:
        df_new = pd.DataFrame(data=X_scaled, index=index, columns=[x for x in range(0,800)])
    else:
        print("HELLO WORLD IT WORKED")
    
    if num == 1:
        pred = lasso(df,df_new,3,0,m)
    elif num == 2:
        pred = ridge(df,df_new,3,0,m)
    elif num == 3:
        pred = linear(df,df_new,3,0,m)
    elif num == 4:
        pred = elasticnet(df,df_new,3,0,m)
    elif num == 5:
        pred = extratree(df,df_new,3,0,m)
    elif num == 6:
        pred = gradientboosting(df,df_new,3,0,m)
    elif num == 7:
        pred = randomforest(df,df_new,3,0,m)
    elif num == 8:
        pred = bayesianridge(df,df_new,3,0,m)
    elif num == 9:
        pred = xgboost(df,df_new,3,0,m)
    
    del df,df_new
    
    pred = list(pred)
    index = list(index)
    
    if len(pred)==len(index):
        with open("results_final_test_data.csv", "w") as f:
            print ("id,loss",file=f)
            for i in range(0,len(pred)):
                print((str(index[i])+','+str(pred[i])),file=f)
                
def lasso(train,test,fs,n,m):
    if fs == 0:
        model = Lasso(alpha=1.0,random_state=0).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
    
        return result
    if fs == 2:
        model = Lasso(alpha=1.0,random_state=0)
        rfe = RFE(model, n_features_to_select=n)
        i = rfe.fit(train.iloc[:,:m], train['loss'])
        predicted = i.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
        
        return result
    if fs == 3:
        model = Lasso(alpha=1.0,random_state=0).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
                  
        return predicted
    else:
        return 0
    
def ridge(train,test,fs,n,m):
    if fs == 0:
        model = Ridge(alpha=1.0,random_state=0).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
    
        return result
    if fs == 2:
        model = Ridge(alpha=1.0,random_state=0)
        rfe = RFE(model, n_features_to_select=n)
        i = rfe.fit(train.iloc[:,:m], train['loss'])
        predicted = i.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
        
        return result
    if fs == 3:
        model = Ridge(alpha=1.0,random_state=0).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
                  
        return predicted
    else:
        return 0
    
def linear(train,test,fs,n,m):
    if fs == 0:
        model = LinearRegression(n_jobs=-1).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
    
        return result
    if fs == 2:
        model = LinearRegression(n_jobs=-1)
        rfe = RFE(model, n_features_to_select=n)
        i = rfe.fit(train.iloc[:,:m], train['loss'])
        predicted = i.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
    
        return result
    if fs == 3:
        model = LinearRegression(n_jobs=-1).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
                  
        return predicted
    else:
        return 0
    
def elasticnet(train,test,fs,n,m):
    if fs == 0:
        model = ElasticNet(alpha=1.0,random_state=0).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
    
        return result
    if fs == 2:
        model = ElasticNet(alpha=1.0,random_state=0)
        rfe = RFE(model, n_features_to_select=n)
        i = rfe.fit(train.iloc[:,:m], train['loss'])
        predicted = i.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
        
        return result
    if fs == 3:
        model = ElasticNet(alpha=1.0,random_state=0).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
                  
        return predicted
    else:
        return 0
    
def extratree(train,test,fs,n,m):
    if fs == 0:
        model = ExtraTreesRegressor(n_jobs=-1,random_state=0).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
    
        return result
    if fs == 2:
        model = ExtraTreesRegressor(n_jobs=-1,random_state=0)
        rfe = RFE(model, n_features_to_select=n)
        i = rfe.fit(train.iloc[:,:m], train['loss'])
        predicted = i.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
        
        return result
    if fs == 3:
        model = ElasticNet(alpha=1.0,random_state=0).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
                  
        return predicted
    else:
        return 0
        
def gradientboosting(train,test,fs,n,m):
    if fs == 0:
        model = GradientBoostingRegressor(random_state=0).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
        
        return result
    if fs == 2:
        model = GradientBoostingRegressor(random_state=0)
        rfe = RFE(model, n_features_to_select=n)
        i = rfe.fit(train.iloc[:,:m], train['loss'])
        predicted = i.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
        
        return result
    if fs == 3:
        model = GradientBoostingRegressor(random_state=0).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
        
        return predicted
    if fs == 47:
        model = GradientBoostingRegressor(random_state=0).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
        
        print ("The MAE is :-",result)
        
        pred = list(predicted)
        index = list(test.index)
        act_labels = list(test['loss'])
        print ("")
        print ("THE PROGRAM WILL GENERATE TWO FILES:-")
        print ("1) File with only id,loss which is the expected(RESULTS_FINAL_test_SUBSET_data(ONLY_ID_AND_LOSS).csv)")
        print ("2) File with id,loss,actual which is an extra csv file just for comparison(RESULTS_COMPARISON_WITH_ACTUAL_LOSS.csv)")
        print ("(PROCESSING...)")
        print ("")
        if len(pred)==len(index)==len(act_labels):
            with open("RESULTS_COMPARISON_WITH_ACTUAL_LOSS.csv", "w") as f:
                print ("id,loss,actual",file=f)
                for i in range(0,len(pred)):
                    print((str(index[i])+','+str(round(pred[i],2))+','+str(act_labels[i])),file=f)
        
        if len(pred)==len(index):
            with open("RESULTS_FINAL_test_SUBSET_data(ONLY_ID_AND_LOSS).csv", "w") as f:
                print ("id,loss",file=f)
                for i in range(0,len(pred)):
                    print((str(index[i])+','+str(round(pred[i],2))),file=f)
    else:
        return 0
    
def randomforest(train,test,fs,n,m):
    if fs == 0:
        model = RandomForestRegressor(n_jobs=-1,n_estimators=m,random_state=0).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
    
        return result
    if fs == 2:
        model = RandomForestRegressor(n_jobs=-1,n_estimators=m,random_state=0)
        rfe = RFE(model, n_features_to_select=n)
        i = rfe.fit(train.iloc[:,:m], train['loss'])
        predicted = i.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
        
        return result
    if fs == 3:
        model = RandomForestRegressor(n_jobs=-1,n_estimators=m,random_state=0).fit(train.iloc[:,:m], train['loss'])
        predicted = clf_5.predict(test.iloc[:,:m])
                  
        return predicted
    else:
        return 0
    
def bayesianridge(train,test,fs,n,m):
    if fs == 0:
        model = BayesianRidge(normalize = True).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
    
        return result
    if fs == 2:
        model = BayesianRidge(normalize = True)
        rfe = RFE(model, n_features_to_select=n)
        i = rfe.fit(train.iloc[:,:m], train['loss'])
        predicted = i.predict(test.iloc[:,:m])
        result = mean_absolute_error(test['loss'], predicted)
        
        return result
    if fs == 3:
        model = BayesianRidge(normalize = True).fit(train.iloc[:,:m], train['loss'])
        predicted = model.predict(test.iloc[:,:m])
                  
        return predicted
    else:
        return 0
    
###def xgboost(train,test,fs,n,m):
    ###if fs == 0:
        ###model = XGBRegressor(n_estimators=60,seed=0,nthread=10).fit(train.iloc[:,:m], train['loss'])
        ###predicted = model.predict(test.iloc[:,:m])
        ###result = mean_absolute_error(test['loss'], predicted)
    
        ###return result
    ###if fs == 3:
        ###model = XGBRegressor(n_estimators=60,seed=0,nthread=10).fit(train.iloc[:,:m], train['loss'])
        ###predicted = model.predict(test.iloc[:,:m])
                  
        ###return predicted
    ###else:
        ###return 0
        
if __name__ == '__main__':
    main()