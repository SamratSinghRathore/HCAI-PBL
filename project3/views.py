from django.shortcuts import render
from palmerpenguins import load_penguins
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

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


from django.shortcuts import render
from palmerpenguins import load_penguins
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
from django.conf import settings

def decision_tree(request):
    try:
        # Get sparsity parameter from request
        ccp_alpha = float(request.GET.get('lambda', 0.01))  # default to 0.01
        
        # Load and prepare data
        penguins = load_penguins()
        penguins = penguins.dropna()
        
        # Prepare features and target
        y = penguins['species']
        X = penguins.drop(['species'], axis=1)
        
        # Encode categorical variables
        le = LabelEncoder()
        X['sex'] = le.fit_transform(X['sex'].astype(str))
        X['island'] = le.fit_transform(X['island'].astype(str))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Initialize and train the model
        model = DecisionTreeClassifier(
            ccp_alpha=ccp_alpha,  # Cost complexity pruning parameter
            random_state=42,
            max_depth=5  # Limit depth for visualization clarity
        )
        
        print("Training model...")
        model.fit(X_train, y_train)

        # Calculate metrics
        print("Calculating metrics...")
        accuracy = model.score(X_test, y_test)
        n_leaves = model.get_n_leaves()
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Number of leaves: {n_leaves}")
        
        # Create visualization
        print("Creating visualization...")
        plt.figure(figsize=(20, 12))
        plt.title(f'Palmer Penguins Decision Tree (α={ccp_alpha})', pad=20, size=16)
        
        from sklearn.tree import plot_tree
        plot_tree(model, 
                 feature_names=X.columns,
                 class_names=model.classes_,
                 filled=True,
                 rounded=True,
                 fontsize=10)

        # Save visualization
        print("Saving visualization...")
        static_dir = os.path.join(settings.BASE_DIR, 'static', 'project3')
        os.makedirs(static_dir, exist_ok=True)
        plt.savefig(os.path.join(static_dir, 'decision_tree.png'), 
                   dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

        print("Process complete!")
        
        context = {
            'accuracy': round(accuracy * 100, 2),
            'n_leaves': n_leaves,
            'lambda_param': ccp_alpha,  # Keep the same parameter name for template compatibility
            'success': True,
            'error_message': None
        }
        
    except Exception as e:
        context = {
            'success': False,
            'error_message': str(e)
        }
    
    return render(request, 'project3/decision_tree.html', context)


from django.shortcuts import render
from palmerpenguins import load_penguins
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
from django.conf import settings

def logistic_regression(request):
    try:
        # Load and prepare data
        penguins = load_penguins()
        penguins = penguins.dropna()
        
        # Get all available features (excluding target)
        all_features = [col for col in penguins.columns if col != 'species']
        
        # Get selected features from POST request or use all features by default
        if request.method == 'POST':
            selected_features = request.POST.getlist('selected_features')
            if not selected_features:  # If nothing selected, use all features
                selected_features = all_features
        else:
            selected_features = all_features
        
        # Prepare features and target
        y = penguins['species']
        X = penguins[selected_features]
        
        # Encode categorical variables if they are selected
        le = LabelEncoder()
        if 'sex' in selected_features:
            X['sex'] = le.fit_transform(X['sex'].astype(str))
        if 'island' in selected_features:
            X['island'] = le.fit_transform(X['island'].astype(str))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train logistic regression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Get accuracy
        accuracy = model.score(X_test, y_test)
        
        # Create visualization of coefficients
        plt.figure(figsize=(12, 6))
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': abs(model.coef_[0])
        })
        feature_importance = feature_importance.sort_values('importance', ascending=True)
        
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance in Logistic Regression')
        plt.xlabel('Absolute Coefficient Value')
        
        # Save the plot
        static_dir = os.path.join(settings.BASE_DIR, 'static', 'project3')
        os.makedirs(static_dir, exist_ok=True)
        plt.savefig(os.path.join(static_dir, 'logistic_regression.png'), 
                   dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

        context = {
            'features': all_features,
            'selected_features': selected_features,
            'accuracy': round(accuracy * 100, 2),
            'success': True,
            'error_message': None,
            'image_url': '/static/project3/logistic_regression.png'
        }
        
    except Exception as e:
        context = {
            'features': all_features if 'all_features' in locals() else [],
            'selected_features': [],
            'success': False,
            'error_message': str(e)
        }
    
    return render(request, 'project3/logistic_regression.html', context)