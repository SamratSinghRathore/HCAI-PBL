from django.shortcuts import render
from palmerpenguins import load_penguins
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

import dice_ml
from dice_ml import Dice
from dice_ml.utils import helpers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
import random
import numpy as np

def index(request):
    # Load data
    penguins = load_penguins()
    penguins_sample = penguins.head(10)
    
    # Get value counts for categorical variables
    value_counts = {
        'species': penguins['species'].value_counts(),
        'island': penguins['island'].value_counts(),
        'sex': penguins['sex'].value_counts(),
        'year': penguins['year'].value_counts()
    }
    
    # Calculate null values for each column
    null_counts = penguins.isnull().sum()
    null_percentages = (null_counts / len(penguins) * 100).round(2)
    null_df = pd.DataFrame({
        'Null Count': null_counts,
        'Null Percentage': null_percentages
    })
    
    # Convert null info to HTML table
    null_table = null_df.to_html(
        classes=['table', 'table-hover', 'table-striped'],
        float_format=lambda x: '{:.2f}%'.format(x) if pd.notnull(x) else '0.00%'
    )
    
    # Rest of your existing code...
    value_counts_html = {}
    for key, counts in value_counts.items():
        value_counts_html[f'{key}_table'] = counts.to_frame().to_html(
            classes=['table', 'table-hover', 'table-striped'],
            header=True,
            index=True
        )
    
    # Convert sample data to HTML
    sample_table = penguins_sample.to_html(
        classes=['table', 'table-hover', 'table-striped'],
        float_format=lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '',
        na_rep='N/A',
        index=False,
        justify='left'
    )
    
    context = {
        'table': sample_table,
        'total_rows': len(penguins),
        'total_columns': len(penguins.columns),
        'displayed_rows': len(penguins_sample),
        'columns': list(penguins.columns),
        'null_table': null_table,
        **value_counts_html
    }
    
    return render(request, 'project3/index.html', context)