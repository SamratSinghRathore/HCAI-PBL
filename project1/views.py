from django.shortcuts import render

# Create your views here.

import csv
import io
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


from django.conf import settings
from django.shortcuts import render
from .forms import CSVUploadForm
from django.http import HttpResponse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from django.shortcuts import render
from .forms import MLForm

def index(request):
    return render(request, 'project1/index.html')


import pandas as pd

import pandas as pd
from django.core.files.storage import FileSystemStorage
from django.shortcuts import redirect

def upload_csv(request):
    form = CSVUploadForm()
    error = None

    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            try:
                # Save the file temporarily
                fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads')) # we use Django's FileSystemStorage to save it into a folder MEDIA_ROOT/uploads
                filename = fs.save(file.name, file)
                request.session['uploaded_csv'] = os.path.join('uploads', filename)
                # return redirect('project1:plot')  # URL name for generate_plot
                action = request.POST.get('action')
                if action == 'plot':
                    return redirect('project1:plot')
                elif action == 'train':
                    return redirect('project1:train_model')
            except Exception as e:
                error = f"Error saving file: {str(e)}"

    return render(request, 'project1/upload.html', {
        'form': form,
        'error': error,
    })



def train_model(request):
    if request.method == "POST":
        form = MLForm(request.POST)
        if form.is_valid():
            try:
                uploaded_path = request.session.get('uploaded_csv')
                if not uploaded_path:
                    return HttpResponse("No file uploaded.")
                # Get the dataset (you might want to modify this part based on how data is managed)
                file_path = os.path.join(settings.MEDIA_ROOT, uploaded_path)
                if not os.path.exists(file_path):
                    return HttpResponse("Uploaded file not found.")
                target_column = form.cleaned_data['target_column_name']
                df = pd.read_csv(file_path)
                if target_column:
                    X = df.drop(target_column, axis=1)
                    y = df[target_column]
                else:
                    X = df.iloc[:,:-1]
                    y = df.iloc[:,-1:]
                
                # Split the dataset
                test_size = form.cleaned_data['test_size']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

                # Select model
                model_choice = form.cleaned_data['model']
                model = None
                if model_choice == 'logreg':
                    model = LogisticRegression()
                    C = form.cleaned_data.get('C', 1.0)
                    if C:
                        model.set_params(C=C)
                elif model_choice == 'dtree':
                    model = DecisionTreeClassifier()
                    max_depth = form.cleaned_data.get('max_depth', None)
                    if max_depth:
                        model.set_params(max_depth=max_depth)
                elif model_choice == 'svm':
                    model = SVC()

                # Train the model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Get scoring metric
                scoring_metric = form.cleaned_data['scoring_metric']
                if scoring_metric == 'accuracy':
                    score = accuracy_score(y_test, y_pred)
                elif scoring_metric == 'precision':
                    score = precision_score(y_test, y_pred, average='macro')  # or 'macro', etc.
                elif scoring_metric == 'recall':
                    score = recall_score(y_test, y_pred, average='macro')  # or 'macro', etc.
                elif scoring_metric == 'f1':
                    score = f1_score(y_test, y_pred, average='macro')  # or 'macro', etc.
                
                # Return result
                context = {
                    'score': round(score, 4),
                    'model': model_choice.upper(),
                    'scoring_metric': scoring_metric.capitalize(),
                }
                return render(request, 'project1/train_result.html', context)
            # except ValueError:
            #     return HttpResponse("The Training data columns should contain only numbers.")
            except Exception as e:
                return HttpResponse(e)

    else:
        form = MLForm()

    return render(request, 'project1/training_form.html', {'form': form})


def generate_plot(request):
    uploaded_path = request.session.get('uploaded_csv')
    if not uploaded_path:
        return HttpResponse("No file uploaded.")

    file_path = os.path.join(settings.MEDIA_ROOT, uploaded_path)
    if not os.path.exists(file_path):
        return HttpResponse("Uploaded file not found.")

    try:
        df = pd.read_csv(file_path)
        required_cols = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'}
        filename = 'iris_plot.png'
        image_path = os.path.join(settings.MEDIA_ROOT, filename)
        if required_cols.issubset(df.columns):

            # Create a 2x1 grid of subplots (2 rows, 1 column)
            fig, axes = plt.subplots(2, 1, figsize=(10, 11))  # 2 rows, 1 column
            fig.tight_layout(pad=3.0)  # Add space between the plots

            # Plot Sepal Length vs Sepal Width on the first subplot
            for species in df['species'].unique():
                subset = df[df['species'] == species]
                axes[0].scatter(subset['sepal_length'], subset['sepal_width'], label=f"{species}", alpha=0.6)

            axes[0].set_xlabel('Sepal Length (cm)')
            axes[0].set_ylabel('Sepal Width (cm)')
            axes[0].set_title('Sepal Length vs Sepal Width by Species')
            axes[0].legend()

            # Plot Petal Length vs Petal Width on the second subplot
            for species in df['species'].unique():
                subset = df[df['species'] == species]
                axes[1].scatter(subset['petal_length'], subset['petal_width'], label=f"{species}", alpha=0.6)

            axes[1].set_xlabel('Petal Length (cm)')
            axes[1].set_ylabel('Petal Width (cm)')
            axes[1].set_title('Petal Length vs Petal Width by Species')
            axes[1].legend()

            # Save the plot to a file
            plt.savefig(image_path)
            plt.close()

            # Generate the URL for the saved image
            image_url = settings.MEDIA_URL + filename

            # Return the image URL to be displayed on the next page
            return render(request, 'project1/show_plot.html', {'image_url': image_url})
        else:
            plt.figure(figsize=(15, 20))  # 2 rows, 1 column
            for col in df.select_dtypes(include='object').columns:
                sns.countplot(y=col, data=df, order=df[col].value_counts().index[:10])
                plt.title(f"Countplot of {col}")
            plt.tight_layout()
            # Save the plot to a file
            plt.savefig(image_path)
            plt.close()

            # Generate the URL for the saved image
            image_url = settings.MEDIA_URL + filename

            # Return the image URL to be displayed on the next page
            return render(request, 'project1/show_plot.html', {'image_url': image_url})



    except Exception as e:
        return HttpResponse(f"Error generating plot: {str(e)}")
