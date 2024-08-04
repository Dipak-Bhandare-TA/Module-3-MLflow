import os 
import warnings
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

import mlflow 
import mlflow.sklearn

from sklearn.tree import DecisionTreeRegressor


def load_data(train_data_path , test_data_path):
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    train_x = train_data.drop('median_house_value' , axis = 1)
    train_y = train_data['median_house_value']

    test_x = train_data.drop('median_house_value' , axis = 1)
    test_y = train_data['median_house_value']

    return train_x, train_y , test_x , test_y



def eval_metric(actual , pred):
    rmse = np.sqrt(mean_squared_error(actual , pred)) 
    r2 = r2_score(actual , pred)
    return rmse , r2


if __name__ =="__main__":


    train_data_path = 'train.csv'
    test_data_path = 'test.csv'

    train_x , train_y , test_x , test_y = load_data(train_data_path  , test_data_path)
    
    max_depth = float(sys.argv[1]) if len(sys.argv) > 1 else None

    tree_model  = DecisionTreeRegressor(max_depth= max_depth )
    
    tree_model.fit(train_x , train_y)
    pred = tree_model.predict(train_x)

    rmse , r2 = eval_metric(train_y , pred)
    
    

    features = train_x.columns
    feature_importnace = tree_model.feature_importances_
    feature_importnace_tab = pd.DataFrame({"feature" : features , "Importance" : feature_importnace})
    feature_importance_df = feature_importnace_tab.sort_values(by="Importance", ascending=False) 

    with mlflow.start_run():

        mlflow.log_param("max_depth" , max_depth)
        
        mlflow.log_metric("RMSe_loss" , rmse)
        mlflow.log_metric("R2_score" , r2)
        
        print("model runs succefully:")
        print("max_depth of decision tree: " , max_depth)
        print("rmse_loss : " , rmse)
        print("r2_score: " , r2) 

        
        mlflow.sklearn.log_model(tree_model, "model")
        
        feature_importance_file_path =  os.path.join(os.getcwd() , "feature_importance_depth:{}.csv".format(max_depth))
        feature_importance_df.to_csv(feature_importance_file_path, index =False)
        mlflow.log_artifact(feature_importance_csv_path)
    
