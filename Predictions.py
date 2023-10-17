import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
dataset= pd.read_csv('Video_Games.csv')
#dataset.isnull().sum()

dt_copy= dataset.copy()

#Check to see if we have same rows
duplicated_rows = dt_copy.duplicated().sum() 

#Change tbd to nan
dt_copy["User_Score"] = dt_copy["User_Score"].replace("tbd", np.nan).astype(float)

#Change User_Score object to float64
dt_copy["User_Score"] = pd.to_numeric(dt_copy.User_Score, errors="coerce")

#Change Publisher's nan values to its most repetitive value (Electronic Arts)
dt_copy.Publisher.value_counts(normalize=True)
dt_copy.Publisher = dt_copy.Publisher.fillna(dt_copy.Publisher.mode()[0])

#Change Rating's nan values to its most repetitive value (E)
#dt_copy.Rating.value_counts(normalize=True)
#dt_copy.Rating = dt_copy.Rating.fillna(dt_copy.Rating.mode()[0])

dt_droppedNaN = dataset.copy()
dt_droppedNaN.dropna(axis="rows", how="any", inplace=True)

#Fill missing values with mean
dt_copy['User_Score'] = dt_copy['User_Score'].fillna(dt_copy['User_Score'].mean())
dt_copy['Critic_Score'] = dt_copy['Critic_Score'].fillna(dt_copy['Critic_Score'].mean())
dt_copy['User_Count'] = dt_copy['User_Count'].fillna(dt_copy['User_Count'].mean())
dt_copy['Critic_Count'] = dt_copy['Critic_Count'].fillna(dt_copy['Critic_Count'].mean())
dt_copy['Year_of_Release'] = dt_copy['Year_of_Release'].fillna(dt_copy['Year_of_Release'].mean())
    
#Change User_Score object to int32
dt_copy["Year_of_Release"] = dt_copy["Year_of_Release"].astype("int32")
    
#Drop nan Genre
dt_copy.dropna(subset=['Genre'],inplace=True)

#Drop useless columns
dt_copy.drop(columns=['Name','Developer','Rating'],axis=1, inplace=True)
    
#dt_copy.isnull().sum()
#dt_copy.info()


while True:
    sales = int(input("Enter number for sales(1,2,3,4,5): "))
                        ############### NA SALES #################
    if sales == 1:   
        label_encoder=LabelEncoder()
        categorical=["Platform", "Genre","Publisher"]
        for cat in categorical:
            dt_copy[cat] = label_encoder.fit_transform(dt_copy[cat])

        #INDEPENDENT
        X = dt_copy[['Platform', 'Genre','Publisher','NA_Sales','Critic_Score','Critic_Count','User_Score','User_Count']].values
        #DEPENDENT
        y = dt_copy['Global_Sales'].values
        onehot_encoder=ColumnTransformer([("Genre", OneHotEncoder(),[1])], remainder='passthrough')
        X=onehot_encoder.fit_transform(X)

        #Splitting data 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
        
        #Feature scaling
        st_x=StandardScaler()
        X_train=st_x.fit_transform(X_train)
        X_test=st_x.transform(X_test)
        
                 
        ############# LINEAR REGRESSION ##########
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred_lin_reg = regressor.predict(X_test)

        
        print('Linear Regression:')
        r2_linear_reg = r2_score(y_test,y_pred_lin_reg)
        print('R2_Score:' , r2_linear_reg)
        mse_lin_reg = mean_squared_error(y_test, y_pred_lin_reg)
        print('MSE:', mse_lin_reg)
        
        ############# DECISION TREE REGRESSION ##########
        dtr = DecisionTreeRegressor(random_state=0)
        dtr.fit(X_train, y_train)
        
        print('\nDecision Tree: ')
        pred_dec_tree = dtr.predict(X_test)
        r2_dec_tree=r2_score(y_test, pred_dec_tree)
        print('R2_Score:',r2_dec_tree )
        mse_dec_tree = mean_squared_error(y_test, pred_dec_tree)
        print('MSE:', mse_dec_tree)
        
        ########### XGB REGRESSION ########
        model = XGBRegressor(n_estimators = 200, learning_rate= 0.08)
        model.fit(X_train, y_train)
        y_pred_xgb = model.predict(X_test)
        
        print('\nXGB: ')
        r2_XGB=r2_score(y_test, y_pred_xgb)
        print('R2_Score:', r2_XGB)
        mse_xgb = mean_squared_error(y_test, y_pred_xgb)
        print('MSE:', mse_xgb)
        
        ############ SVM REGRESSION ##########
        svm = SVR(C = 10, gamma=0.1)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)

        print('\nSVR: ')
        r2_SVM=r2_score(y_test, y_pred_svm)
        print('R2_Score:', r2_SVM)
        mse_svm = mean_squared_error(y_test, y_pred_svm)
        print('MSE:', mse_svm)
        
        ######### RANDOM FOREST REGRESSION ##########
        from sklearn import metrics

        random_forest = RandomForestRegressor(max_depth=50,random_state=60,min_samples_split=3,max_samples=0.2)
        random_forest.fit(X_train, y_train)
        y_pred_rf = random_forest.predict(X_test)
          
        print('\nRandom Forest: ')
        print('R2_Score:',metrics.r2_score(y_test, y_pred_rf))
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rf))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))
        print('Max Error:',metrics.max_error(y_test, y_pred_rf))
        print('Variance: ', np.std(y_pred_rf))
        print('Standard deviation: ', np.sqrt(np.std(y_pred_rf)))


        ######### KNN REGRESSION #############
        knn = KNeighborsRegressor(n_neighbors=10)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        
        print('\nKNN: ')
        r2_KNN = r2_score(y_test, y_pred_knn)
        print('R2_Score:',r2_KNN)
        mse_knn = mean_squared_error(y_test, y_pred_knn)
        print('MSE:', mse_knn)
        
        ############ RIDGE REGRESSION ##############
        ridge = Ridge(alpha=10)
        ridge.fit(X_train, y_train)
        y_pred_rd = ridge.predict(X_test)
        
        print('\nRidge: ')
        r2_rd = r2_score(y_test, y_pred_rd)
        print('R2_Score:', r2_rd)
        mse_rd = mean_squared_error(y_test, y_pred_rd)
        print('MSE:', mse_rd)
        
        ########### LASSO REGRESSION ############
        lasso = Lasso(alpha=0.10)
        lasso.fit(X_train, y_train)
        y_pred_ls = lasso.predict(X_test)
            
        print('\nLasso: ')
        r2_ls= r2_score(y_test, y_pred_ls)
        print('R2_Score:',r2_ls )
        mse_ls = mean_squared_error(y_test, y_pred_ls)
        print('MSE:', mse_ls)
        
        
    
                ################  EU SALES  ##################    
    elif sales==2:
        label_encoder=LabelEncoder()
        categorical=["Platform", "Genre","Publisher"]
        for cat in categorical:
            dt_copy[cat] = label_encoder.fit_transform(dt_copy[cat])

        #INDEPENDENT
        X = dt_copy[['Platform', 'Genre','Publisher','EU_Sales','Critic_Score','Critic_Count','User_Score','User_Count']].values
        #DEPENDENT
        y = dt_copy['Global_Sales'].values
        onehot_encoder=ColumnTransformer([("Genre", OneHotEncoder(),[1])], remainder='passthrough')
        X=onehot_encoder.fit_transform(X)

        #Splitting data 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

        #Feature scaling
        st_x=StandardScaler()
        X_train=st_x.fit_transform(X_train)
        X_test=st_x.transform(X_test)

        ############# LINEAR REGRESSION ##########
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred_lin_reg = regressor.predict(X_test)

        print('Linear Regression:')
        r2_linear_reg = r2_score(y_test,y_pred_lin_reg)
        print('R2_Score:' , r2_linear_reg)
        mse_lin_reg = mean_squared_error(y_test, y_pred_lin_reg)
        print('MSE:', mse_lin_reg)
        
        ############# DECISION TREE REGRESSION ##########
        dtr = DecisionTreeRegressor(random_state=0)
        dtr.fit(X_train, y_train)
        
        print('\nDecision Tree: ')
        pred_dec_tree = dtr.predict(X_test)
        r2_dec_tree=r2_score(y_test, pred_dec_tree)
        print('R2_Score:',r2_dec_tree )
        mse_dec_tree = mean_squared_error(y_test, pred_dec_tree)
        print('MSE:', mse_dec_tree)
            
        
        ########### XGB REGRESSION #########
        mdl_eu = XGBRegressor(random_state=60)
        mdl_eu.fit(X_train, y_train)

        eu_kfold = KFold(n_splits=8)

        print('\nXGB:')     
        xgb_eu_r2 = cross_val_score(mdl_eu, X_train, y_train, cv=eu_kfold,scoring='r2')
        xgb_eu_mae=cross_val_score(mdl_eu, X_train, y_train, cv=eu_kfold,scoring='neg_mean_squared_error')
        xgb_eu_mabe=cross_val_score(mdl_eu, X_train, y_train, cv=eu_kfold,scoring='neg_mean_absolute_error')
        xgb_eu_mabpe=cross_val_score(mdl_eu, X_train, y_train, cv=eu_kfold,scoring='neg_mean_absolute_percentage_error')

        print("R2_Score: %0.2f Standard Deviation: %0.2f" %(xgb_eu_r2.mean(), xgb_eu_r2.std()))
        print("Negative Mean Squared Error: %0.2f Standard Deviation: %0.2f" %(xgb_eu_mae.mean(), xgb_eu_mae.std()))
        print("Negative Mean Absolute Error: %0.2f Standard Deviation: %0.2f" %(xgb_eu_mabe.mean(), xgb_eu_mabe.std()))
        print("Negative Mean Absolute Percentage Error: %0.2f Standard Deviation: %0.2f" %(xgb_eu_mabpe.mean(), xgb_eu_mabpe.std()))

        
# =============================================================================
#         parameters = {
#             'max_depth': range (2, 10, 1),
#             'learning_rate': [0.01, 0.05,0.08]
#             }
#        
# # Initialize XGB and GridSearch
#         xgb = XGBRegressor(nthread=-1,n_estimators=200) 
#         grid = GridSearchCV(xgb,parameters,cv=5)
#         grid.fit(X_train, y_train)
# 
# 
#         print('\nXGB:')
#         print('R2 Score: ', r2_score(y_test, grid.best_estimator_.predict(X_test))) 
#         
# =============================================================================
        
        ############ SVM REGRESSION ##########
        svm = SVR(C = 10, gamma=0.1)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        
        print('\nSVR: ')
        r2_SVM=r2_score(y_test, y_pred_svm)
        print('R2_Score:', r2_SVM)
        mse_svm = mean_squared_error(y_test, y_pred_svm)
        print('MSE:', mse_svm)
        
        ######### RANDOM FOREST REGRESSION ##########
        random_forest = RandomForestRegressor(random_state=60)
        random_forest.fit(X_train, y_train)
        y_pred_rf = random_forest.predict(X_test)
        
        print('\nRandom Forest: ')
        r2_rf = r2_score(y_test, y_pred_rf)
        print('R2_Score:',r2_rf )
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        print('MSE:', mse_rf)
        
        ######### KNN REGRESSION #############
        knn = KNeighborsRegressor(n_neighbors=10)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
    
        print('\nKNN: ')
        r2_KNN = r2_score(y_test, y_pred_knn)
        print('R2_Score:',r2_KNN)
        mse_knn = mean_squared_error(y_test, y_pred_knn)
        print('MSE:', mse_knn)
        
        ############ RIDGE REGRESSION ##############
        ridge = Ridge(alpha=10)
        ridge.fit(X_train, y_train)
        y_pred_rd = ridge.predict(X_test)
        
        print('\nRidge: ')
        r2_rd = r2_score(y_test, y_pred_rd)
        print('R2_Score:', r2_rd)
        mse_rd = mean_squared_error(y_test, y_pred_rd)
        print('MSE:', mse_rd)
        
        ########### LASSO REGRESSION ############
        lasso = Lasso(alpha=0.10)
        lasso.fit(X_train, y_train)
        y_pred_ls = lasso.predict(X_test)
            
        print('\nLasso: ')
        r2_ls= r2_score(y_test, y_pred_ls)
        print('R2_Score:',r2_ls )
        mse_ls = mean_squared_error(y_test, y_pred_ls)
        print('MSE:', mse_ls)
        

        
                    ################# JP SALES ################
    elif sales==3:    
        label_encoder=LabelEncoder()
        categorical=["Platform", "Genre","Publisher"]
        for cat in categorical:
            dt_copy[cat] = label_encoder.fit_transform(dt_copy[cat])
            
            #INDEPENDENT
        X = dt_copy[['Platform', 'Genre','Publisher','JP_Sales','Critic_Score','Critic_Count','User_Score','User_Count']].values
            #DEPENDENT
        y = dt_copy['Global_Sales'].values
        onehot_encoder=ColumnTransformer([("Genre", OneHotEncoder(),[1])], remainder='passthrough')
        X=onehot_encoder.fit_transform(X)

        #Splitting data 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
        
        #Feature scaling
        st_x=StandardScaler()
        X_train=st_x.fit_transform(X_train)
        X_test=st_x.transform(X_test)
        

        ############# LINEAR REGRESSION ##########
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred_lin_reg = regressor.predict(X_test)

        
        print('Linear Regression:')
        r2_linear_reg = r2_score(y_test,y_pred_lin_reg)
        print('R2_Score:' , r2_linear_reg)
        mse_lin_reg = mean_squared_error(y_test, y_pred_lin_reg)
        print('MSE:', mse_lin_reg)
        
        ############# DECISION TREE REGRESSION ##########
        dtr = DecisionTreeRegressor(random_state=0)
        dtr.fit(X_train, y_train)
        
        print('\nDecision Tree: ')
        pred_dec_tree = dtr.predict(X_test)
        r2_dec_tree=r2_score(y_test, pred_dec_tree)
        print('R2_Score:',r2_dec_tree )
        mse_dec_tree = mean_squared_error(y_test, pred_dec_tree)
        print('MSE:', mse_dec_tree)
        
        ########### XGB REGRESSION #########
        model = XGBRegressor(n_estimators = 200, learning_rate= 0.08)
        model.fit(X_train, y_train)
        y_pred_xgb = model.predict(X_test)
        
        print('\nXGB: ')
        r2_XGB=r2_score(y_test, y_pred_xgb)
        print('R2_Score:', r2_XGB)
        mse_xgb = mean_squared_error(y_test, y_pred_xgb)
        print('MSE:', mse_xgb)
        
        ############ SVM REGRESSION ##########
        svm = SVR(C = 10, gamma=0.1)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        
        print('\nSVR: ')
        r2_SVM=r2_score(y_test, y_pred_svm)
        print('R2_Score:', r2_SVM)
        mse_svm = mean_squared_error(y_test, y_pred_svm)
        print('MSE:', mse_svm)
        
        ######### RANDOM FOREST REGRESSION ##########        

        rf_jp = RandomForestRegressor(random_state=30)
        rf_jp.fit(X_train, y_train)
        folds_jp = KFold(n_splits = 6)

        rf_jp_r2 = cross_val_score(rf_jp, X_train, y_train, cv=folds_jp,scoring='r2')
        rf_jp_mae=cross_val_score(rf_jp, X_train, y_train, cv=folds_jp,scoring='neg_mean_squared_error')
        rf_jp_mabe=cross_val_score(rf_jp, X_train, y_train, cv=folds_jp,scoring='neg_mean_absolute_error')
        rf_jp_mabpe=cross_val_score(rf_jp, X_train, y_train, cv=folds_jp,scoring='neg_mean_absolute_percentage_error')
        print('\nRandom Forest: ')
        print("R2_Score: %0.2f Standard Deviation: %0.2f" %(rf_jp_r2.mean(), rf_jp_r2.std()))
        print("Negative Mean Squared Error: %0.2f Standard Deviation: %0.2f" %(rf_jp_mae.mean(), rf_jp_mae.std()))
        print("Negative Mean Absolute Error: %0.2f Standard Deviation: %0.2f" %(rf_jp_mabe.mean(), rf_jp_mabe.std()))
        print("Negative Mean Absolute Percentage Error: %0.2f Standard Deviation: %0.2f" %(rf_jp_mabpe.mean(), rf_jp_mabpe.std()))

        
# =============================================================================
#       rf_kf = RandomForestRegressor(random_state=30)
#       params = {
#             "max_depth": [4,5,8],
#             "n_estimators": [100,200,500,1000]
#             }
#         
#       model_cv = GridSearchCV(rf_kf, params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
#       print(model_cv.best_params_)
#       print(model_cv.best_score_)
#       Random Forest: 
#       {'max_depth': 8, 'n_estimators': 500}
#       0.606094902029629
#
# =============================================================================
        
        ######### KNN REGRESSION #############
        knn = KNeighborsRegressor(n_neighbors=10)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        
        print('\nKNN: ')
        r2_KNN = r2_score(y_test, y_pred_knn)
        print('R2_Score:',r2_KNN)
        mse_knn = mean_squared_error(y_test, y_pred_knn)
        print('MSE:', mse_knn)
        
        ############ RIDGE REGRESSION ##############
        ridge = Ridge(alpha=10)
        ridge.fit(X_train, y_train)
        y_pred_rd = ridge.predict(X_test)
        
        print('\nRidge: ')
        r2_rd = r2_score(y_test, y_pred_rd)
        print('R2_Score:', r2_rd)
        mse_rd = mean_squared_error(y_test, y_pred_rd)
        print('MSE:', mse_rd)
        
        ########### LASSO REGRESSION ############
        lasso = Lasso(alpha=0.10)
        lasso.fit(X_train, y_train)
        y_pred_ls = lasso.predict(X_test)
            
        print('\nLasso: ')
        r2_ls= r2_score(y_test, y_pred_ls)
        print('R2_Score:',r2_ls )
        mse_ls = mean_squared_error(y_test, y_pred_ls)
        print('MSE:', mse_ls)
        


                ############### OTHER SALES ##############
    elif sales==4:
        label_encoder=LabelEncoder()
        categorical=["Platform", "Genre","Publisher"]
        for cat in categorical:
            dt_copy[cat] = label_encoder.fit_transform(dt_copy[cat])

        #INDEPENDENT
            X = dt_copy[['Platform', 'Genre','Publisher','Other_Sales','Critic_Score','Critic_Count','User_Score','User_Count']].values
        #DEPENDENT
            y = dt_copy['Global_Sales'].values
            onehot_encoder=ColumnTransformer([("Genre", OneHotEncoder(),[1])], remainder='passthrough')
            X=onehot_encoder.fit_transform(X)
        
        #Splitting data 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
        
        #Feature scaling
        st_x=StandardScaler()
        X_train=st_x.fit_transform(X_train)
        X_test=st_x.transform(X_test)


        ############# LINEAR REGRESSION ##########
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred_lin_reg = regressor.predict(X_test)

        
        print('Linear Regression:')
        r2_linear_reg = r2_score(y_test,y_pred_lin_reg)
        print('R2_Score:' , r2_linear_reg)
        mse_lin_reg = mean_squared_error(y_test, y_pred_lin_reg)
        print('MSE:', mse_lin_reg)
    
        ############# DECISION TREE REGRESSION ##########
        dtr = DecisionTreeRegressor(random_state=0)
        dtr.fit(X_train, y_train)
        
        print('\nDecision Tree: ')
        pred_dec_tree = dtr.predict(X_test)
        r2_dec_tree=r2_score(y_test, pred_dec_tree)
        print('R2_Score:',r2_dec_tree )
        mse_dec_tree = mean_squared_error(y_test, pred_dec_tree)
        print('MSE:', mse_dec_tree)
            
        ########### XGB REGRESSION #########
        model_oth = XGBRegressor(random_state=30)
        model_oth.fit(X_train, y_train)
        oth_fold = KFold(n_splits = 8)
        
        print('\nXGB: ')         
        oth_cv_r2 = cross_val_score(model_oth, X_train, y_train, cv=oth_fold,scoring='r2')
        oth_cv_mae=cross_val_score(model_oth, X_train, y_train, cv=oth_fold,scoring='neg_mean_squared_error')
        oth_cv_mabe=cross_val_score(model_oth, X_train, y_train, cv=oth_fold,scoring='neg_mean_absolute_error')
        oth_cv_mabpe=cross_val_score(model_oth, X_train, y_train, cv=oth_fold,scoring='neg_mean_absolute_percentage_error')

        print("R2_Score: %0.2f Standard Deviation: %f" %(oth_cv_r2.mean(), oth_cv_r2.std()))
        print("Negative Mean Squared Error: %f Standard Deviation: %f" %(oth_cv_mae.mean(), oth_cv_mae.std()))
        print("Negative Mean Absolute Error: %f Standard Deviation: %f" %(oth_cv_mabe.mean(), oth_cv_mabe.std()))
        print("Negative Mean Absolute Percentage Error: %f Standard Deviation: %f" %(oth_cv_mabpe.mean(), oth_cv_mabpe.std()))
        
        ############ SVM REGRESSION ##########
        svm = SVR(C = 10, gamma=0.1)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        
        print('\nSVR: ')
        r2_SVM=r2_score(y_test, y_pred_svm)
        print('R2_Score:', r2_SVM)
        mse_svm = mean_squared_error(y_test, y_pred_svm)
        print('MSE:', mse_svm)
        
        ######### RANDOM FOREST REGRESSION ##########
        random_forest = RandomForestRegressor(random_state=60)
        random_forest.fit(X_train, y_train)
        y_pred_rf = random_forest.predict(X_test)
        
        print('\nRandom Forest: ')
        r2_rf = r2_score(y_test, y_pred_rf)
        print('R2_Score:',r2_rf )
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        print('MSE:', mse_rf)
        
        ######### KNN REGRESSION #############
        knn = KNeighborsRegressor(n_neighbors=10)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        
        print('\nKNN: ')
        r2_KNN = r2_score(y_test, y_pred_knn)
        print('R2_Score:',r2_KNN)
        mse_knn = mean_squared_error(y_test, y_pred_knn)
        print('MSE:', mse_knn)
        
        ############ RIDGE REGRESSION ##############
        ridge = Ridge(alpha=10)
        ridge.fit(X_train, y_train)
        y_pred_rd = ridge.predict(X_test)
        
        print('\nRidge: ')
        r2_rd = r2_score(y_test, y_pred_rd)
        print('R2_Score:', r2_rd)
        mse_rd = mean_squared_error(y_test, y_pred_rd)
        print('MSE:', mse_rd)

        ########### LASSO REGRESSION ############
        lasso = Lasso(alpha=0.10)
        lasso.fit(X_train, y_train)
        y_pred_ls = lasso.predict(X_test)
            
        print('\nLasso: ')
        r2_ls= r2_score(y_test, y_pred_ls)
        print('R2_Score:',r2_ls )
        mse_ls = mean_squared_error(y_test, y_pred_ls)
        print('MSE:', mse_ls)
        

                ################  WITHOUT SALES  ##################    
    elif sales==5:
        label_encoder = LabelEncoder()
        categorical = ["Platform", "Genre", "Publisher"]
        for cat in categorical:
            dt_droppedNaN[cat] = label_encoder.fit_transform(dt_droppedNaN[cat])

        # INDEPENDENT
        X = dt_droppedNaN[['Platform', 'Genre', 'Publisher', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']].values
        #DEPENDENT
        y = dt_droppedNaN['Global_Sales'].values
        onehot_encoder = ColumnTransformer([("Genre", OneHotEncoder(), [1])], remainder='passthrough')
        X = onehot_encoder.fit_transform(X)

        #Splitting data 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

        #Feature scaling
        st_x=StandardScaler()
        X_train=st_x.fit_transform(X_train)
        X_test=st_x.transform(X_test)

        ############# LINEAR REGRESSION ##########
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred_lin_reg = regressor.predict(X_test)

        print('Linear Regression:')
        r2_linear_reg = r2_score(y_test,y_pred_lin_reg)
        print('R2_Score:' , r2_linear_reg)
        mse_lin_reg = mean_squared_error(y_test, y_pred_lin_reg)
        print('MSE:', mse_lin_reg)
    
        ############# DECISION TREE REGRESSION ##########
        dtr = DecisionTreeRegressor(random_state=30)
        dtr.fit(X_train, y_train)
        
        print('\nDecision Tree: ')
        pred_dec_tree = dtr.predict(X_test)
        r2_dec_tree=r2_score(y_test, pred_dec_tree)
        print('R2_Score:',r2_dec_tree )
        mse_dec_tree = mean_squared_error(y_test, pred_dec_tree)
        print('MSE:', mse_dec_tree)
            
        ########### XGB REGRESSION #########
        from sklearn import metrics


        mdl_wth = XGBRegressor(n_estimators = 700, learning_rate= 0.08)
        mdl_wth.fit(X_train, y_train)
        y_pred_wth = mdl_wth.predict(X_test)
        print('\nXGB: ')

        print('R2_Score:',metrics.r2_score(y_test, y_pred_wth))
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_wth))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_wth))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_wth)))
        print('Variance: ', np.std(y_pred_wth))
        print('Standard deviation: ', np.sqrt(np.std(y_pred_wth)))

        
        ############ SVM REGRESSION ##########
        svm = SVR(C = 10, gamma=0.1)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        
        print('\nSVR: ')
        r2_SVM=r2_score(y_test, y_pred_svm)
        print('R2_Score:', r2_SVM)
        mse_svm = mean_squared_error(y_test, y_pred_svm)
        print('MSE:', mse_svm)
        
        ######### RANDOM FOREST REGRESSION ##########
        random_forest = RandomForestRegressor(random_state=60)
        random_forest.fit(X_train, y_train)
        y_pred_rf = random_forest.predict(X_test)
        
        print('\nRandom Forest: ')
        r2_rf = r2_score(y_test, y_pred_rf)
        print('R2_Score:',r2_rf )
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        print('MSE:', mse_rf)
        
        ######### KNN REGRESSION #############
        knn = KNeighborsRegressor(n_neighbors=10)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        
        print('\nKNN: ')
        r2_KNN = r2_score(y_test, y_pred_knn)
        print('R2_Score:',r2_KNN)
        mse_knn = mean_squared_error(y_test, y_pred_knn)
        print('MSE:', mse_knn)
        
        ############ RIDGE REGRESSION ##############
        ridge = Ridge(alpha=10)
        ridge.fit(X_train, y_train)
        y_pred_rd = ridge.predict(X_test)
        
        print('\nRidge: ')
        r2_rd = r2_score(y_test, y_pred_rd)
        print('R2_Score:', r2_rd)
        mse_rd = mean_squared_error(y_test, y_pred_rd)
        print('MSE:', mse_rd)

        ########### LASSO REGRESSION ############
        lasso = Lasso(alpha=0.10)
        lasso.fit(X_train, y_train)
        y_pred_ls = lasso.predict(X_test)
            
        print('\nLasso: ')
        r2_ls= r2_score(y_test, y_pred_ls)
        print('R2_Score:',r2_ls )
        mse_ls = mean_squared_error(y_test, y_pred_ls)
        print('MSE:', mse_ls)
        

    else:
        print("Error!!! Try again")
       