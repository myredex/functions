# Functions that will help to solve classic ML problems using Sklearn Pipeline

# Table of contents:
# get_transformed_column_names - returns list of the columns after preprocessing in the pipeline

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
