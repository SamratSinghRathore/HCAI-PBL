from django import forms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

class CSVUploadForm(forms.Form):
    file = forms.FileField(label='Select a CSV file')

class MLForm(forms.Form):
    model_choices = [
        ('logreg', 'Logistic Regression'),
        ('dtree', 'Decision Tree'),
        ('svm', 'SVM'),
    ]
    
    scoring_choices = [
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1 Score'),
    ]

    model = forms.ChoiceField(choices=model_choices)
    test_size = forms.FloatField(label="Test set size", initial=0.3, min_value=0.0, max_value=1.0)
    scoring_metric = forms.ChoiceField(choices=scoring_choices, initial='accuracy')
    target_column_name = forms.CharField(label="Target column name(Default: Last column)", empty_value="", required=False)

    # For specific hyperparameters, such as max_depth for decision tree, etc.
    max_depth = forms.IntegerField(label="Max Depth (for Decision Tree)", required=False)
    C = forms.FloatField(label="C (for Logistic Regression)", required=False)
