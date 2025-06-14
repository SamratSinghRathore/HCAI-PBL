from django.shortcuts import render
from palmerpenguins import load_penguins
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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


def decision_tree(request):
    # Load data
    penguins = load_penguins()
    penguins = penguins.dropna()
    
    # Prepare features and target
    y = penguins['species']
    X = penguins.drop('species', axis=1)
    
    # Convert categorical variables
    le = LabelEncoder()
    X['sex'] = le.fit_transform(X['sex'])
    X['island'] = le.fit_transform(X['island'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train decision tree
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Get accuracy and number of leaves
    accuracy = clf.score(X_test, y_test)
    n_leaves = clf.get_n_leaves()
    
    context = {
        'accuracy': round(accuracy * 100, 2),
        'n_leaves': n_leaves,
    }
    
    return render(request, 'project3/decision_tree.html', context)