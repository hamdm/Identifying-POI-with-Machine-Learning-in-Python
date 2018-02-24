def featureSelector(dataframe,target,features,selectors):
    '''
    This function takes in a model based and iterative feature selection technique, 
    and returns the score and selected features against each feature selection technique
    '''
    target = dataframe[target]
    features = dataframe.loc[:,features]
    xtrain,xtest,ytrain,ytest = train_test_split(features,target,random_state=100)
    lgR = LogisticRegression()
    score_feature = {}
    
    for tech in selectors:
        print(tech)
        print(type(tech))
        if tech=='SelectFromModel':
            select = SelectFromModel(RandomForestClassifier())
        elif tech=='RFE':
            select = RFE(RandomForestClassifier(),n_features_to_select=12)
    
        select.fit(xtrain,ytrain)
        transformed_xtrain = select.transform(xtrain)
        transformed_xtest = select.transform(xtest)
        support = select.get_support()
        bestFeatures=list(xtrain.columns[support])
        score = lgR.fit(transformed_xtrain,ytrain).score(transformed_xtest,ytest)
        score_feature[tech] = [score,bestFeatures]
    return score_feature