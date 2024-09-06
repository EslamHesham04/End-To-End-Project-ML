import pandas as pd
import os
## sklearn -- for pipeline and preprocessing
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder # type: ignore
from sklearn.pipeline import Pipeline, FeatureUnion # type: ignore
from sklearn_features.transformers import DataFrameSelector # type: ignore


## Read the CSV file using pandas
df = pd.read_csv("C:\\Users\\Eslam\\Desktop\\End-To-End-ML-Project\\hr_employee_churn_data.csv")


## Try to make some Feature Engineering --> Feature Extraction --> Add the new column to the main DF
df['number_project_per_time_spend_company']=df["number_project"]/df["time_spend_company"] # type: ignore
df["satisfaction_level_per_last_evaluation"]=df["satisfaction_level"]/df["last_evaluation"] # type: ignore
df["satisfaction_level_per_number_project"]=df["satisfaction_level"]/df["number_project"] # type: ignore


## Split the Dataset -- Taking only train to fit (the same the model was trained on)
X = df.drop(columns=['left','empid'], axis=1)  ## features
y = df['left']  ## target

## the same Random_state (take care)
X_train , X_test, y_train , y_test = train_test_split(X , y, test_size=0.3, random_state=123, shuffle=True)

## Separete the columns according to type (numerical or categorical)
num_cols = [col for col in  X_train.columns if X_train[col].dtype in ['float64','float32','int32' ,'int64']]
categ_cols = [col for col in  X_train.columns if X_train[col].dtype not in ['float64','float32','int32' ,'int64']]

## We can get much much easier like the following
## numerical pipeline
num_pipeline = Pipeline([
                    ('selector', DataFrameSelector(num_cols)),    
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())])

categ_pipeline = Pipeline(steps=[
            ('selector', DataFrameSelector(categ_cols)),    
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('OHE', OneHotEncoder(sparse_output=False))])

total_pipeline = FeatureUnion(transformer_list=[
                                ('num_pip', num_pipeline),
                                ('categ_pipeline', categ_pipeline)])


X_train_final = total_pipeline.fit_transform(X_train)

def preprocess_new(X_new):
    ''' This Function tries to process the new instances before predicted using Model
    Args:
    *****
        (X_new: 2D array) --> The Features in the same order
                ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 
                 'population', 'households', 'median_income', 'ocean_proximity']
        All Featutes are Numerical, except the last one is Categorical.
        
     Returns:
     *******
         Preprocessed Features ready to make inference by the Model
    '''
    return total_pipeline.transform(X_new)