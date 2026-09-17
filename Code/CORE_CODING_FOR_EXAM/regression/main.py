import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score


def read_data():
    train_path = '/Users/filippomontecchi/Desktop/Data Science & Machine Learning Lab/Lab/Dataset/LAB5/train_dataset.csv'
    test_path = '/Users/filippomontecchi/Desktop/Data Science & Machine Learning Lab/Lab/Dataset/LAB5/test_dataset.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    x_train = train_df.loc[:, 'cont_0':'cat_7']
    y_train = train_df['target']

    x_test = test_df.loc[:, 'cont_0':'cat_7']

    return x_train, y_train, x_test


def drop_useless():
    x_train.drop(columns=['cat_4', 'cat_5', 'cat_6'], inplace=True)
    x_test.drop(columns=['cat_4', 'cat_5', 'cat_6'], inplace=True)


def preprocessing_training(x_train):
    num_col = x_train.loc[:, 'cont_0': 'cont_29']
    ord_col = x_train.loc[:, 'ord_0': 'ord_19']
    cat_col = x_train.loc[:, 'cat_0':'cat_7']

    num_col_names = ['cont_0', 'cont_1', 'cont_2', 'cont_3', 'cont_4', 'cont_5', 'cont_6',
        'cont_7', 'cont_8', 'cont_9', 'cont_10', 'cont_11', 'cont_12',
        'cont_13', 'cont_14', 'cont_15', 'cont_16', 'cont_17', 'cont_18',
        'cont_19', 'cont_20', 'cont_21', 'cont_22', 'cont_23', 'cont_24',
        'cont_25', 'cont_26', 'cont_27', 'cont_28', 'cont_29']
    ord_col_names = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'ord_6', 'ord_7',
        'ord_8', 'ord_9', 'ord_10', 'ord_11', 'ord_12', 'ord_13', 'ord_14',
        'ord_15', 'ord_16', 'ord_17', 'ord_18', 'ord_19']
    cat_col_names = ['cat_0', 'cat_1', 'cat_2', 'cat_3', 'cat_7']


    all_ord = []
    for o_c in ord_col:
        # print(ord_col[o_c].value_counts().head(5))
        # print(ord_col[o_c].value_counts().tail(10))
        non_nan_ord_values_per_col = ord_col[o_c].dropna()
        unique_ord_values = non_nan_ord_values_per_col.unique().tolist()
        sorted_unique_ord_values = sorted(unique_ord_values, key=lambda x:x.split('_')[3])
        all_ord.append(sorted_unique_ord_values)
    



    num_pipe = Pipeline(steps=[
        (
            'num_imputer',
            SimpleImputer(strategy='median', add_indicator=True)
        )
        ,
        (
            'scaler',
            StandardScaler()
        )
    ])

    ord_pipe = Pipeline(steps=[
        (
            'ord_imputer',
            SimpleImputer(strategy='constant', fill_value='unknown')    # add_indicator = True --> if you explicitly pass categories it might fuck up everything
        )
        ,
        (
            'OE',
            OrdinalEncoder(categories=all_ord, handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-2)      # Nan becomes -1 while missing ordinal values -2 : eg row 5 hasn't got ord_7_val_1 --> puts value -2
        )
    ])

    cat_pipe = Pipeline(steps=[
        (
            'cat_imputer',
            SimpleImputer(strategy='constant', fill_value='unknown', add_indicator=True)
        )
        ,
        (
            'OHE',
            OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        )
    ])

    preprocessing_pipeline = ColumnTransformer(transformers=[
        ('num_prep', num_pipe, num_col_names),
        ('ord_prep', ord_pipe, ord_col_names),
        ('cat_prep', cat_pipe, cat_col_names)
    ], remainder='passthrough', n_jobs=-1)

    full_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('regression', RandomForestRegressor(
            n_estimators = 300,
            max_depth = None,
            min_samples_split = 5,
            min_samples_leaf = 5,
            n_jobs=-1,
            random_state=42)
        )
    ])
    
    full_pipeline.fit(x_train, y_train)
    
    y_pred_train = full_pipeline.predict(x_train)
    print(f'training score: {r2_score(y_train, y_pred_train)}')

    return full_pipeline


def prediction(x_test, full_pipeline):
    y_pred = full_pipeline.predict(x_test)
    
    y_pred_df = pd.DataFrame({'ID': range(len(y_pred)), 'predictions': y_pred})
    y_pred_df.to_csv('predictions.csv', index=False)


if __name__ == '__main__':
    x_train, y_train, x_test = read_data()
    drop_useless()
    full_pipeline = preprocessing_training(x_train)

    prediction(x_test, full_pipeline)
