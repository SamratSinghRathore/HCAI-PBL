# views.py
from django.shortcuts import render
import pandas as pd

def index(request):
    csv_files = [
        ('Links',   r'project4\ml-latest-small\links.csv'),
        ('Movies',  r'project4\ml-latest-small\movies.csv'),
        ('Ratings', r'project4\ml-latest-small\ratings.csv'),
        ('Tags',    r'project4\ml-latest-small\tags.csv'),
    ]

    tables = []
    for name, path in csv_files:
        try:
            # ▸ only grab the first 5 rows
            df = pd.read_csv(path, nrows=5)
            dimensions = pd.read_csv(path).shape
            tables.append({
                'dimensions': dimensions,
                'name':  name,
                'table': df.to_html(
                    classes=['table', 'table-hover', 'table-striped'],
                    index=False
                )
            })
        except Exception as e:
            tables.append({
                'dimensions': (0, 0),
                'name':  name,
                'table': f"<p style='color:red;'>Error loading {name}: {e}</p>"
            })

    return render(request, 'project4/index.html', {'tables': tables,})
