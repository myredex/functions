# Functions that will help to solve classic ML problems using Sklearn Pipeline

# Table of contents:
# class CustomColumnTransformer - class to use with CoLumnTransformer performs basic manipulations with columns
# get_transformed_column_names - returns list of the columns after preprocessing in the pipeline
# 


# =========================================================
import math
import pandas as pd

class CustomColumnTransformer(BaseEstimator, TransformerMixin):
    """
    This class perform basic math transformations with columns of the dataframe
    inside the pipeline. All you need is to set lists of columns and 
    mode type to transform the columns.

    Args:
        columns: list of lists of the columns to transform
        mode: 'multiply', 'divide', 'subtract', 'sum'

    Example:
        custom_preprocessor = Pipeline([
            ('imputer', CustomColumnTransformer(columns = [['float1', 'float2']], 
            mode='subtract'))
            ])

    Returns:
        Dataframe with transformed columns
    """
    
    def __init__(self, columns: list[list], mode: str):

        # Validate columns is list of lists
        if not all(isinstance(i, list) for i in columns):
            raise ValueError("Columns must be list of lists")

        # Validate mode type
        if mode not in ['multiply', 'divide', 'subtract', 'sum']:
            raise ValueError("Mode must be str 'multiply' or 'divide'")

        self.columns = columns
        self.mode = mode
        self.new_column_names = []


    def fit(self, X, y=None):
        return self
    

    def transform(self, X):
        X.copy() # Copy original dataframe
        dataframe = pd.DataFrame() # Empty list of dataframes

        # Run preprocessing for each sublist
        for list_of_columns in self.columns:
            if self.mode == 'multiply':
                new_column_name = str('multiply_' + '_'.join(list_of_columns))
                dataframe[new_column_name] = X[list_of_columns].prod(axis=1)

            elif self.mode == 'divide':
                new_column_name = str('divide_' + '_'.join(list_of_columns))
                dataframe[new_column_name] = X[list_of_columns].iloc[:, 0]

                for col in list_of_columns[1:]:
                    dataframe[new_column_name] /= X[col]

            elif self.mode == 'subtract':
                new_column_name = str('subtract_' + '_'.join(list_of_columns))
                dataframe[new_column_name] = X[list_of_columns].iloc[:, 0]

                for col in list_of_columns[1:]:
                    dataframe[new_column_name] /= X[col]

            elif self.mode == 'sum':
                new_column_name = str('sum_' + '_'.join(list_of_columns))
                dataframe[new_column_name] = X[list_of_columns].sum(axis=1)          
        
        self.new_column_names = dataframe.columns
        return dataframe
    
    def get_feature_names_out(self, input_features=None):
        """
        Returns names of the columns
        """

        return self.new_column_names


# =========================================================

def get_transformed_column_names(pipeline, preprocessor):
    """
    Function returns list of the columns after preprocessing in the
    pipeline
    Args:
        :param pipeline: Sklearn Pipeline 
        :param preprocessor: Name of the ColumnTransformer's preprocessor

    Returns: 
        list of the names of the dataframe after preprocessing

    Example:
        get_transformed_column_names(pipeline, 'preprocessor')
    """
    
    # Get dict with the names of colum transformers
    named_transformers = pipeline.named_steps[preprocessor].named_transformers_
    all_col_names = []  # Define empty list

    for key, value in named_transformers.items(): # Run throught the dict

        # If only one preprocessor there is 'reminder' in the dict dont include it
        if key != 'remainder':
            # Get names of the columns and collect it in list
            col_names = pipeline.named_steps[preprocessor].named_transformers_[key].get_feature_names_out()
            all_col_names.extend(col_names)

    return all_col_names
