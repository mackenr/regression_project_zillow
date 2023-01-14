from wrangle import *
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor,ElasticNet,LassoLarsIC,BayesianRidge,ARDRegression 
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score








def  linearmodel(X_train,X_validate,y_train,y_validate,degree=1):
    try:
        # Prep for the Polynomial model

        #  make the polynomial features to get a new set of features
        pf = PolynomialFeatures(degree=degree)

        # fit and transform X_train_scaled
        X_train = pf.fit_transform(X_train)

        # transform X_validate_scaled 
        X_validate = pf.transform(X_validate)

        # create the model object
        lm = LinearRegression(normalize=True)

        # fit the model to our training data. We must specify the column in y_train, since we have converted it to a dataframe from a series! 
        lm.fit(X_train, y_train.iloc[:, 0])

        # predict train
        y_train['taxvaluedollarcnt_pred_lm'] = lm.predict(X_train)
        # evaluate: rmse
        rmse_train = mean_squared_error(y_train.iloc[:, 0], y_train.taxvaluedollarcnt_pred_lm)**(1/2)

        # predict validate
        y_validate['taxvaluedollarcnt_pred_lm'] = lm.predict(X_validate)
        # evaluate: rmse
        rmse_validate = mean_squared_error(y_validate.iloc[:, 0], y_validate.taxvaluedollarcnt_pred_lm)**(1/2)

        
        model_name=(f'{lm.__class__.__name__} degree {degree}')
        model_params=lm.get_params(deep=False)
        model= lm 
        model_tupple=(model_name,model_params,degree,rmse_train,rmse_validate,model)
        
        return model_tupple
        # model_list.append(lm)
    except:
        print(f'error with {lm.__class__.__name__}')




def bestTest(X_train, y_train, X_validate, y_validate, X_test, y_test,best):
    '''
    this is created simply to minimize the code in the main juoyternotebook 
    
    '''
    # rmse_list=[]
    # model_name_list=[]

   
    


    ## fit if the model is linear
    if  np.all(best['Degree']!='N/A'):
        print('linear\n')
        degree=best['Degree'].values[0]    
         ## Prep for the degree of linear model
        #  make the polynomial features to get a new set of features
        pf = PolynomialFeatures(degree=degree)
        # fit and transform X_train_scaled
        X_train= pf.fit_transform(X_train)
        # transform X_validate_scaled
        X_validate = pf.transform(X_validate)
        # transform X_validate_scaled 
        X_test = pf.transform(X_test)
    else:
        print('non linear\n')
        



    ## RMSE for the 3rd Degree Polynomial Model 


    # recover the model object
    
    rmseDF=full_model(X_train,X_validate,y_train,y_validate,X_test, y_test,best)
    return rmseDF



def full_model(X_train,X_validate,y_train,y_validate,X_test, y_test,best):
    model=best['model_obj'].values[0]
    degree=best['Degree'].values[0]
    try:
        model.fit(X_train, y_train.iloc[:, 0])
        # predict train
        y_train['taxvaluedollarcnt_pred'] = model.predict(X_train)
        rmse_train = mean_squared_error(y_train.iloc[:, 0], y_train.taxvaluedollarcnt_pred)**(1/2)
        # predict validate
        y_validate['taxvaluedollarcnt_pred'] = model.predict(X_validate)        
        rmse_validate = mean_squared_error(y_validate.iloc[:, 0], y_validate.taxvaluedollarcnt_pred)**(1/2)
        # predict test
        y_test['taxvaluedollarcnt_pred'] = model.predict(X_test)    
        rmse_test= mean_squared_error(y_test.iloc[:,0], y_test.taxvaluedollarcnt_pred)**(1/2)


        if degree!='N/A':
            model_name=f'{model.__class__.__name__} Degree: {degree}'
        else:
            model_name=model.__class__.__name__

        model_params=model.get_params(deep=False)

        modellist=[]
        model_tupple=(model_name,(model_params),degree,rmse_train,rmse_validate,rmse_test)
        modellist.append(model_tupple)
        cols=['model_name','model_params','degree','Train','Validate','Test']

        rmseDF=pd.DataFrame(data=modellist,columns=cols)
        rmseDF['Train_to_Val_diff']=rmseDF.Train-rmseDF.Validate
        rmseDF['Train_to_Val_abs_diff']=rmseDF['Train_to_Val_diff'].apply(abs)
        rmseDF['Train_to_Test_diff']=rmseDF.Train-rmseDF.Test
        rmseDF['Train_to_Test_abs_diff']=rmseDF['Train_to_Test_diff'].apply(abs)

        #
        plt.figure(figsize=(16,8))
        plt.scatter(y_train.iloc[:,0], y_train.taxvaluedollarcnt_pred,alpha=.5, color="yellow", s=100, label=f"Model: {model_name} | Train")
        plt.scatter(y_validate.iloc[:,0], y_validate.taxvaluedollarcnt_pred,alpha=.5, color="green", s=100, label=f"Model: {model_name} | Validate")
        plt.scatter(y_test.iloc[:,0], y_test.taxvaluedollarcnt_pred,alpha=.5, color="red", s=100, label=f"Model: {model_name} | Test")
        plt.legend()
        plt.xlabel("Actual Tax value")
        plt.ylabel("Predicted Actual Tax value")
        plt.title("Best Model")
        plt.savefig(fname=f'TrainValTestGraph{model_name}.png' ,dpi='figure',format='png')
        plt.show()
        return rmseDF
    except:
        print(f'error with {model.__class__.__name__}')

        
       
        # model_name_not_base.append('RMSE for Tweedie, power=1 & alpha=0')
        # model_list.append(glm)








def nonlinearmodel(X_train,X_validate,y_train,y_validate,model=TweedieRegressor(power=1.5, alpha=0.25)):
    try:
        # create the model object   
        #  fit the model to our training data. We must specify the column in y_train, 
        # since we have converted it to a dataframe from a series! 
        model.fit(X_train, y_train.iloc[:, 0])
        # predict train
        y_train['taxvaluedollarcnt_pred_glm'] = model.predict(X_train)
        # evaluate: rmse
        rmse_train = mean_squared_error(y_train.iloc[:, 0], y_train.taxvaluedollarcnt_pred_glm)**(1/2)
        # predict validate
        y_validate['taxvaluedollarcnt_pred_glm'] = model.predict(X_validate)
        # evaluate: rmse
        rmse_validate = mean_squared_error(y_validate.iloc[:, 0], y_validate.taxvaluedollarcnt_pred_glm)**(1/2)
        # print(f'''RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample:  {rmse_train:.4g}
        #       "\nValidate/Out-of-Sample: \n {rmse_validate:.4g}''')
        model_name=(f'{model.__class__.__name__}')
        model_params=model.get_params(deep=False)
        degree='N/A'
        model_tupple=(model_name,model_params,degree,rmse_train,rmse_validate,model)

        return model_tupple
    except:
        print(f'error with {model.__class__.__name__}')

        
       
        # model_name_not_base.append('RMSE for Tweedie, power=1 & alpha=0')
        # model_list.append(glm)




def baseline_rmse(y_train,y_validate):
    'using mean, median, and mode finds the lowest abs difference rmse score and returns that as baseline'
    model_list=[]
   
   

    # 1. Predict taxvaluedollarcnt_pred_mean & Calc RMSE
    taxvaluedollarcnt_pred_mean = y_train.iloc[:, 0].mean()
    y_train['taxvaluedollarcnt_pred_mean'] = taxvaluedollarcnt_pred_mean
    y_validate['taxvaluedollarcnt_pred_mean'] = taxvaluedollarcnt_pred_mean
    rmse_train = mean_squared_error(y_train.iloc[:, 0], y_train.taxvaluedollarcnt_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.iloc[:, 0], y_validate.taxvaluedollarcnt_pred_mean)**(1/2)    
    model_name='Mean'
    degree=model_params=model_obj='N/A'
    model_tupple=(model_name,model_params,degree,rmse_train,rmse_validate,model_obj)
    model_list.append(model_tupple)
    # 2. compute taxvaluedollarcnt_pred_median & Calc RMSE
    taxvaluedollarcnt_pred_median = y_train.iloc[:, 0].median()
    y_train['taxvaluedollarcnt_pred_median'] = taxvaluedollarcnt_pred_median
    y_validate['taxvaluedollarcnt_pred_median'] = taxvaluedollarcnt_pred_median
    rmse_train = mean_squared_error(y_train.iloc[:, 0], y_train.taxvaluedollarcnt_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.iloc[:, 0], y_validate.taxvaluedollarcnt_pred_median)**(1/2)
    model_name='Median'
    degree=model_params=model_obj='N/A'
    model_tupple=(model_name,model_params,degree,rmse_train,rmse_validate,model_obj)
    model_list.append(model_tupple)
    # 3. Compute taxvaluedollarcnt_pred_mode & Calc RMSE
    taxvaluedollarcnt_pred_mode = y_train.iloc[:, 0].mode().values[0]
    y_train['taxvaluedollarcnt_pred_mode'] = taxvaluedollarcnt_pred_mode
    y_validate['taxvaluedollarcnt_pred_mode'] = taxvaluedollarcnt_pred_mode
    rmse_train = mean_squared_error(y_train.iloc[:, 0], y_train.taxvaluedollarcnt_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.iloc[:, 0], y_validate.taxvaluedollarcnt_pred_median)**(1/2)
    model_name='Mode'
    degree=model_params=model_obj='N/A'
    model_tupple=(model_name,model_params,degree,rmse_train,rmse_validate,model_obj)
    model_list.append(model_tupple)
    # model_list
    rmseDF=pd.DataFrame(model_list,columns=['Model[RSME]','Params','Degree','Train','Validate','model_obj'])
    rmseDF['diff']=rmseDF.Train-rmseDF.Validate
    rmseDF['abs_diff']=rmseDF['diff'].apply(abs)
    rmseDF['abs_percent_change']=(rmseDF['abs_diff']/rmseDF.Train)*100
    rmseDF.sort_values(by=['abs_diff'],inplace=True)
    baseline=rmseDF.iloc[[0]]
    baseline['Params']=baseline['Model[RSME]']
    baseline['Model[RSME]']='Baseline'

    return baseline




def regmodelbest(X_train, y_train, X_validate, y_validate, X_test, y_test, random=123):
    '''
    in the future this should be be updated for more automation. 
    https://machinelearninghd.com/gridsearchcv-hyperparameter-tuning-sckit-learn-regression-classification/
    offers some examples
    Currently it allows for the best models to be visualized. I will then apply a gradient search to the best model
    to optimize further

    Started adding a dictionary to pick the best model but other models could be added
    
    
    
    '''

   
    


    # make a list to store model rmse in and dict to pull best model from
    
    model_list=[]



    
    X_train=pd.DataFrame(X_train)
    X_validate=pd.DataFrame(X_validate)
    X_test=pd.DataFrame(X_test)
    cols=select_best(X_train, y_train, 4, model = LinearRegression())

    X_train=X_train[cols]
    X_validate=X_validate[cols]
    X_test=X_test[cols]
   

  

    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    ## adding y_test to be used at the end
    y_test = pd.DataFrame(y_test)

     ## Baseline
    baseline=baseline_rmse(y_train,y_validate)

    

    #4. Linear regessors
    for deg in range(1,5):
        try:
            model_tupple=linearmodel(X_train,X_validate,y_train,y_validate,degree=deg)
            model_list.append(model_tupple)
        except:
            print('error')

    ## RMSE for Elastic Net
    try:
        model_tupple=nonlinearmodel(X_train,X_validate,y_train,y_validate,model=ElasticNet(tol=1e-6))
        model_list.append(model_tupple)
    except:
        print('error')    

    ##  RMSE for LassoLarsIC 

    try:
        model_tupple=nonlinearmodel(X_train,X_validate,y_train,y_validate,model=LassoLarsIC())
        model_list.append(model_tupple)
    except:
        print('error')


    ##  RMSE for ARDRegression     
    try:
        model_tupple=nonlinearmodel(X_train,X_validate,y_train,y_validate,model=ARDRegression())
        model_list.append(model_tupple)
    except:
        print('error')

    ##  RMSE for Lasso + Lars   
    try:
        model_tupple=nonlinearmodel(X_train,X_validate,y_train,y_validate,model=LassoLars(alpha=1.0))
        model_list.append(model_tupple)
    except:
        print('error')

    ## RMSE for BayesianRidge
    try:
        model_tupple=nonlinearmodel(X_train,X_validate,y_train,y_validate,model=BayesianRidge(n_iter=int(1e4),tol=1e-8))
        model_list.append(model_tupple)
    except:
        print('error')

    ##Tweedie 
    try:
        model_tupple=nonlinearmodel(X_train,X_validate,y_train,y_validate,model=TweedieRegressor(power=1, alpha=0))
        model_list.append(model_tupple)
    except:
        print('error')

  
    try:
        model_tupple=nonlinearmodel(X_train,X_validate,y_train,y_validate,model=TweedieRegressor(power=0, alpha=0))
        model_list.append(model_tupple)
    except:
        print('error')
         
         
    try:
        model_tupple=nonlinearmodel(X_train,X_validate,y_train,y_validate,model=TweedieRegressor(power=1.5, alpha=0.25))
        model_list.append(model_tupple)
    except:
        print('error')
         

    # modelsresmedict=dict(zip(model_name_list,rmse_list))
    rmseDF=pd.DataFrame(data=model_list,columns=['Model[RSME]','Params','Degree','Train','Validate','model_obj'])
    rmseDF=pd.concat([baseline,rmseDF],ignore_index=True)
    # rmseDF=rmseDF.T
    rmseDF['diff']=rmseDF.Train-rmseDF.Validate
    rmseDF['abs_diff']=rmseDF['diff'].apply(abs)
    rmseDF['abs_percent_change']=(rmseDF['abs_diff']/rmseDF.Train)*100
    
    rmseDF.sort_values(by=['abs_diff'],inplace=True)
    rmseDF=rmseDF[['Model[RSME]','Params','Degree','Train','Validate','abs_percent_change','model_obj']]

    #islolates the top row our best model for testing
    best=rmseDF.iloc[[0]]
   

    rmseDF_full=bestTest(X_train, y_train, X_validate, y_validate, X_test, y_test,best)

    display('All')
    display(rmseDF)
    print('Best')
    display(best)
    print('Best w/Test')
    display(rmseDF_full)
    # Here we find the average rmse from our train and val for baseline and our best model, then we calculate a percent change to see the improvment from baseline
    base=rmseDF.where(rmseDF['Model[RSME]']=='Baseline').dropna()
    best=rmseDF.iloc[[0]]
    avermseforbase=np.mean([base.Train.values[0],base.Validate.values[0]])
    avermseforbest=np.mean([best.Train.values[0],best.Validate.values[0]])
    percentchange=((avermseforbest-avermseforbase)/avermseforbase)*100
    print(f'Our Average RMSE from % Change from baseline\n{percentchange}')





    # plot to visualize actual vs predicted. 
   
 
    return rmseDF,rmseDF_full





# ============== Feature Selection ====================

def select_kbest(X_train, y_train, k_features):
    '''
    Takes in X_train, y_train, and the number of features to select.
    Returns the names of the selected features using SelectKBest from sklearn. 
    '''
    kbest = SelectKBest(f_regression, k=k_features)
    kbest.fit(X_train, y_train)
    
    return(X_train.columns[kbest.get_support()].tolist())
    
    
def select_rfe(X_train, y_train, k_features, model = LinearRegression()):
    '''
    Takes in X_train, y_train, and the number of features to select.
    Returns the names of the selected features using Recursive Feature Elimination from sklearn. 
    '''
    
    rfe = RFE(model, n_features_to_select=k_features)
    rfe.fit(X_train, y_train)
    
    return(X_train.columns[rfe.support_].tolist())
    
    
def select_sfs(X_train, y_train, k_features, model = LinearRegression()):
    '''
    Takes in X_train, y_train, and the number of features to select.
    Returns the names of the selected features using Sequential Feature Selector from sklearn. 
    '''
    sfs = SequentialFeatureSelector(model, n_features_to_select=k_features)
    sfs.fit(X_train, y_train)
    
    return(X_train.columns[sfs.support_].tolist())

def select_best(X, y_train, k_features, model = LinearRegression()):
    '''
    Takes in X_train, y_train, and the number of features to select.
    Optional Model selection (default: LinearRegression())
    Runs select_kbest, select_rfe, and select sfs functions to find the different bests.
    Returns bests.
    '''
    X=pd.DataFrame(X)
    X_train=X.select_dtypes(include='number')
    kbest=select_kbest(X_train, y_train, k_features)
    rfe=select_rfe(X_train, y_train, k_features, model)
    sfs=select_sfs(X_train, y_train, k_features, model)
    fullset=list(set(kbest)| set(rfe)| set(sfs))
    
    print("KBest:")
    print(kbest)
    print(" ")
    print("RFE:")
    print(rfe)
    print(" ")
    print("SFS:")
    print(sfs)
    print(" ")
    print('Combined Set')
    print(fullset)
    print(" ")
    return fullset








def single_split_many_return(df,totarget='taxvaluedollarcnt'):
    '''
    
  
    returns fulllist,fullscalledlist,lalist,lascaled,oclist,ocscalledlist   
    where they all have their respective X_train, y_train, X_validate, y_validate, X_test, y_test in position 0...5 
    the nonscaled lists have their respective non split test for viz purposes in position 6
    need to make this an function or mayb an object too much code
    
    
    '''


    
    
  
    train, validate, test=train_validate_test(df)
    ##Aggregate
       
    X_train, y_train, X_validate, y_validate, X_test, y_test=bigX_little_y(train,  validate,  test, target=totarget)
    fulllist=[] 
    fulllist.append(X_train)
    fulllist.append(y_train)
    fulllist.append(X_validate)
    fulllist.append(y_validate)
    fulllist.append(X_test)
    fulllist.append(y_test)
    fulllist.append(train) 
   

     
     # make, fit, use:
     ##LA
       
    la_train=train[train.county=='LA']
    la_validate=validate[validate.county=='LA']
    la_test=test[test.county=='LA']
    X_la_train, y_la_train, X_la_validate, y_la_validate, X_la_test, y_la_test=bigX_little_y(la_train,  la_validate,  la_test, target=totarget)
    lalist=[]
    lalist.append(X_la_train)
    lalist.append(y_la_train)
    lalist.append(X_la_validate)
    lalist.append(y_la_validate)
    lalist.append(X_la_test)
    lalist.append(y_la_test)
    lalist.append(la_train)

    ##Orange
   
    orange_train=train[train.county=='Orange']
    orange_validate=validate[validate.county=='Orange']
    orange_test=test[test.county=='Orange']    
    X_orange_train, y_orange_train, X_orange_validate, y_orange_validate, X_orange_test, y_orange_test=bigX_little_y(orange_train,  orange_validate,  orange_test, target=totarget)
    oclist=[]
    oclist.append(X_orange_train)
    oclist.append(y_orange_train)
    oclist.append(X_orange_validate)
    oclist.append(y_orange_validate)
    oclist.append(X_orange_test)
    oclist.append(y_orange_test)
    oclist.append(orange_train)


    ##Ventura
    
    ventura_train=train[train.county=='Ventura']
    ventura_validate=validate[validate.county=='Ventura']
    ventura_test=test[test.county=='Ventura']
    X_ventura_train, y_ventura_train, X_ventura_validate, y_ventura_validate, X_ventura_test, y_ventura_test=bigX_little_y(ventura_train,  ventura_validate,  ventura_test, target=totarget)
    venturalist=[]
    venturalist.append(X_ventura_train)
    venturalist.append(y_ventura_train)
    venturalist.append(X_ventura_validate)
    venturalist.append(y_ventura_validate)
    venturalist.append(X_ventura_test)
    venturalist.append(y_ventura_test)    

    

   



    # make the object, put it into the variable scaler
    scaler = MinMaxScaler()
    train_scaled=train
    validate_scaled=validate
    test_scaled=test
    # fit the object to my data:



    ##LA
    la_train_scaled=train_scaled[train_scaled.county=='LA']
    la_validate_scaled=validate_scaled[validate_scaled.county=='LA']
    la_test_scaled=test_scaled[test_scaled.county=='LA']
    la_train_scaled=la_train_scaled.drop(columns=['county','fips'   ])
    la_validate_scaled=la_validate_scaled.drop(columns=['county','fips'   ])
    la_test_scaled=la_test_scaled.drop(columns=['county','fips'   ])
    la_train_scaled=la_train_scaled.select_dtypes(include='number')
    la_validate_scaled=la_validate_scaled.select_dtypes(include='number')
    la_test_scaled=la_test_scaled.select_dtypes(include='number')
    X_la_train_scaled, y_la_train_scaled, X_la_validate_scaled, y_la_validate_scaled, X_la_test_scaled, y_la_test_scaled=bigX_little_y(la_train_scaled,  la_validate_scaled,  la_test_scaled, target=totarget)
    X_la_train_scaled=scaler.fit_transform(X_la_train_scaled)
    X_la_validate_scaled=scaler.transform(X_la_validate_scaled)
    X_la_test_scaled=scaler.transform(X_la_test_scaled)
    lascaled=[]   
    lascaled.append(X_la_train_scaled)
    lascaled.append(y_la_train_scaled)
    lascaled.append(X_la_validate_scaled)
    lascaled.append(y_la_validate_scaled)
    lascaled.append(X_la_test_scaled)
    lascaled.append(y_la_test_scaled)

    ##Orange
    orange_train_scaled=train_scaled[train_scaled.county=='Orange']
    orange_validate_scaled=validate_scaled[validate_scaled.county=='Orange']
    orange_test_scaled=test_scaled[test_scaled.county=='Orange']
    orange_train_scaled=orange_train_scaled.drop(columns=['county','fips'   ])
    orange_validate_scaled=orange_validate_scaled.drop(columns=['county','fips'   ])
    orange_test_scaled=orange_test_scaled.drop(columns=['county','fips'   ])
    orange_train_scaled=orange_train_scaled.select_dtypes(include='number')
    orange_validate_scaled=orange_validate_scaled.select_dtypes(include='number')
    orange_test_scaled=orange_test_scaled.select_dtypes(include='number')
    X_orange_train_scaled, y_orange_train_scaled, X_orange_validate_scaled, y_orange_validate_scaled, X_orange_test_scaled, y_orange_test_scaled=bigX_little_y(orange_train_scaled,  orange_validate_scaled,  orange_test_scaled, target=totarget)
    X_orange_train_scaled=scaler.fit_transform(X_orange_train_scaled)
    X_orange_validate_scaled=scaler.transform(X_orange_validate_scaled)
    X_orange_test_scaled=scaler.transform(X_orange_test_scaled)
    ocscalledlist=[]
    ocscalledlist.append(X_orange_train_scaled)
    ocscalledlist.append(y_orange_train_scaled)
    ocscalledlist.append(X_orange_validate_scaled)
    ocscalledlist.append(y_orange_validate_scaled)
    ocscalledlist.append(X_orange_test_scaled)
    ocscalledlist.append(y_orange_test_scaled)    


    #Ventura
    ventura_train_scaled=train_scaled[train_scaled.county=='Ventura']
    ventura_validate_scaled=validate_scaled[validate_scaled.county=='Ventura']
    ventura_test_scaled=test_scaled[test_scaled.county=='Ventura']
    ventura_train_scaled=ventura_train_scaled.drop(columns=['county','fips'   ])
    ventura_validate_scaled=ventura_validate_scaled.drop(columns=['county','fips'   ])
    ventura_test_scaled=ventura_test_scaled.drop(columns=['county','fips'   ])   
    ventura_train_scaled=ventura_train_scaled.select_dtypes(include='number')
    ventura_validate_scaled=ventura_validate_scaled.select_dtypes(include='number')
    ventura_test_scaled=ventura_test_scaled.select_dtypes(include='number')
    X_ventura_train_scaled, y_ventura_train_scaled, X_ventura_validate_scaled, y_ventura_validate_scaled, X_ventura_test_scaled, y_ventura_test_scaled=bigX_little_y(ventura_train_scaled,  ventura_validate_scaled,  ventura_test_scaled, target=totarget)
    X_ventura_train_scaled=scaler.fit_transform(X_ventura_train_scaled)
    X_ventura_validate_scaled=scaler.transform(X_ventura_validate_scaled)
    X_ventura_test_scaled=scaler.transform(X_ventura_test_scaled)
    venturascalledlist=[]
    venturascalledlist.append(X_ventura_train_scaled)
    venturascalledlist.append(y_ventura_train_scaled)
    venturascalledlist.append(X_ventura_validate_scaled)
    venturascalledlist.append(y_ventura_validate_scaled)
    venturascalledlist.append(X_ventura_test_scaled)
    venturascalledlist.append(y_ventura_test_scaled)

   
  
    #Aggregate
    train_scaled=train_scaled.drop(columns=['county','fips'   ])
    validate_scaled=validate_scaled.drop(columns=['county','fips'   ])
    test_scaled=test_scaled.drop(columns=['county','fips'   ])
    train_scaled=train_scaled.select_dtypes(include='number')
    validate_scaled=validate_scaled.select_dtypes(include='number')
    test_scaled=test_scaled.select_dtypes(include='number')
    X_scale_train, y_scale_train, X_scale_validate, y_scale_validate, X_scale_test, y_scale_test=bigX_little_y(train_scaled,  validate_scaled,  test_scaled, target=totarget)
    X_scale_train=scaler.fit_transform(X_scale_train)
    X_scale_validate= scaler.transform(X_scale_validate) 
    X_scale_test= scaler.transform(X_scale_test)
    fullscalledlist=[]
    fullscalledlist.append(X_scale_train)
    fullscalledlist.append(y_scale_train)
    fullscalledlist.append(X_scale_validate)
    fullscalledlist.append(y_scale_validate)
    fullscalledlist.append(X_scale_test)
    fullscalledlist.append(y_scale_test)

    # train=pd.merge(X_train, y_train,left_index=True,right_index=True)

    # train_la=pd.merge(X_la_train, y_la_train,left_index=True,right_index=True)

    # orange_train=pd.merge(X_orange_train, y_orange_train,left_index=True,right_index=True)

    # orange_train=pd.merge(X_orange_train, y_orange_train,left_index=True,right_index=True)


    return   fulllist,fullscalledlist,lalist,lascaled,oclist,ocscalledlist,venturalist,venturascalledlist


# ============== Viz ====================
def goldenvisvars(m=1.75):
    '''
    m is our scaler
    created height and width ratios and aspect based on the golden ratio 
    
    
    '''
    h=21**(1/2)
 
    w=21**(1/2)+13**(1/2)
    h=h*m
    w=w*m  
    aspect=(1+(5)**.5)/2
    h_int=int(h*.95)

    return h,w,aspect,h_int















def encodeforZillow(df):
    notencodelist=['area',
        'taxvaluedollarcnt',
        'yearbuilt',
        'parcelid',
        'lotsizesquarefeet',
        'longitude',
        'latitude',
        'transactiondate_in_days',
        'logerror',
        'lotsize/area',
        'bedbath_harmean',
        'bedbath_harmeandividesarea',
        'sqrt(bed^2+bath^2)',
        'sqrt(bed^2+bath^2)dividesarea',
        'bathplusbathdividesarea',
        'sqrt(bed^2+bath^2)divides(lotsize/area)',
        'bedbath_harmean)divides(lotsize/area)',
        'latscaled',
        'longscaled',
        'latlongPythagC',
        'latlongPythagC*(sqrt(bed^2+bath^2)divides(lotsize/area))',
        'lotsizesquarefeet_wo_outliers']
    encodelist=list(set(df.columns)-set(notencodelist))

    df=pd.get_dummies(df, prefix=None, prefix_sep='_', dummy_na=False, columns=encodelist, sparse=False, drop_first=False, dtype=bool)
    return df





