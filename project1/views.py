from django.shortcuts import render

# Create your views here.

import csv
import io
import os
import numpy as np
from matplotlib import pyplot as plt


from django.conf import settings
from django.shortcuts import render
from .forms import CSVUploadForm
from django.http import HttpResponse

def index(request):
    return HttpResponse("Welcome to Project 1!")


import pandas as pd

def upload_csv(request):
    form = CSVUploadForm()
    error = None
    image_url = None

    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']

            try:
                df = pd.read_csv(file)
                print(df.columns)
                # Check required columns exist
                required_cols = {'sepal_length', 'sepal_width', 'species'}
                if not required_cols.issubset(df.columns):
                    error = f"CSV must contain columns: {', '.join(required_cols)}"
                else:
                    # Plotting with seaborn-style colors
                    filename = 'iris_plot.png'
                    image_path = os.path.join(settings.MEDIA_ROOT, filename)

                    plt.figure(figsize=(6, 4))
                    for species in df['species'].unique():
                        subset = df[df['species'] == species]
                        plt.scatter(subset['sepal_length'], subset['sepal_width'], label=species)

                    plt.xlabel('Sepal Length')
                    plt.ylabel('Sepal Width')
                    plt.title('Sepal Length vs Width by Species')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(image_path)
                    plt.close()

                    image_url = settings.MEDIA_URL + filename

            except Exception as e:
                error = f"Error processing file: {str(e)}"

    return render(request, 'project1/upload.html', {
        'form': form,
        'error': error,
        'image_url': image_url
    })




def generate_plot(request):
    filename = 'myplot.png'
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    
    x = np.random.rand(10)
    y = np.random.rand(10)
    plt.scatter(x, y)
    plt.savefig(image_path)
    
    image_url = settings.MEDIA_URL + filename
    return render(request, 'project1/show_plot.html', {'image_url': image_url}) # was demos/show_plot.html