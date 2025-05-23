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
from sklearn.svm import SVC
from django.shortcuts import render
# from .forms import MLForm
from django.shortcuts import render
from django.http import JsonResponse
import json
import time
import pickle
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scipy.sparse import vstack as sparse_vstack
# Temporary global state (in-memory, resets with server restart)
ACTIVE_LEARNING_STATE = {}


def index(request):
    return render(request, 'project2/index.html')


import pandas as pd

import pandas as pd
from django.core.files.storage import FileSystemStorage
from django.shortcuts import redirect

def train_model(request):
    if request.method == 'POST':
        try:

            # Parse the request body for AJAX requests
            data = json.loads(request.body)
            print(data)
            representation = data.get('representation')
            classifier = data.get('classifier')
            
            # Start timing
            start_time = time.time()

            # Loading data
            X_train, X_test, y_train, y_test = load_IMDB_data()
            
            # Initialize the text representation module based on selection
            if representation == 'tfidf':
                vectorizer = TfidfVectorizer(max_features=10000)
                representation_name = "TF-IDF"
            elif representation == 'bow':
                vectorizer = CountVectorizer(max_features=10000)
                representation_name = "Bag of Words"
            else:
                return JsonResponse({'error': 'Invalid representation method'}, status=400)
            
            # Transform the text data
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            
            # Initialize the classifier based on selection
            if classifier == 'logreg':
                model = LogisticRegression(max_iter=1000)
                classifier_name = "Logistic Regression"
            elif classifier == 'nn':
                model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
                classifier_name = "Neural Network"
            else:
                return JsonResponse({'error': 'Invalid classifier method'}, status=400)
            
            # Train the model
            model.fit(X_train_vec, y_train)

            # Evaluate on test set
            accuracy = model.score(X_test_vec, y_test)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Save the model and vectorizer for later use
            model_dir = os.path.join('project2', 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            with open(os.path.join(model_dir, f'vectorizer_{representation}.pkl'), 'wb') as f:
                pickle.dump(vectorizer, f)
            
            with open(os.path.join(model_dir, f'model_{representation}_{classifier}.pkl'), 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata about the current model
            model_metadata = {
                'representation': representation,
                'classifier': classifier,
                'accuracy': accuracy,
                'representation_name': representation_name,
                'classifier_name': classifier_name
            }
            
            with open(os.path.join(model_dir, 'current_model_metadata.json'), 'w') as f:
                json.dump(model_metadata, f)
            
            # Return results to the frontend
            return JsonResponse({
                'accuracy': float(accuracy),  # Convert numpy float to Python float if needed
                'training_time': round(training_time, 2),
                'representation_name': representation_name,
                'classifier_name': classifier_name
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    # If GET request, render the page or return error
    return JsonResponse({'error': 'POST method required'}, status=400)


def load_model(request):
    if request.method == 'POST':
        representation_dict = {
            "bow": "Bag Of Words",
            "tfidf": "TF-IDF",
        }
        
        classifier_dict = {
            "logreg": "Logistic Regression",
            "nn": "Neural Network"
        }
        
        try:
            # Parse the request body for AJAX requests
            data = json.loads(request.body)
            representation = data.get('representation')
            classifier = data.get('classifier')
            
            # Validate inputs
            if not representation or not classifier:
                return JsonResponse({'error': 'Missing representation or classifier parameter'}, status=400)
            
            model_dir = os.path.join('project2', 'models')
            
            # Check if model files exist
            model_path = os.path.join(model_dir, f'model_{representation}_{classifier}.pkl')
            vectorizer_path = os.path.join(model_dir, f'vectorizer_{representation}.pkl')
            
            if not os.path.exists(model_path):
                return JsonResponse({'error': f'Model file not found: {model_path}'}, status=404)
            
            if not os.path.exists(vectorizer_path):
                return JsonResponse({'error': f'Vectorizer file not found: {vectorizer_path}'}, status=404)
            
            # Load the model and vectorizer (FIXED: changed 'wb' to 'rb')
            with open(model_path, 'rb') as f:  # READ mode, not WRITE mode
                loaded_classifier = pickle.load(f)
                
            with open(vectorizer_path, 'rb') as f:  # READ mode, not WRITE mode
                loaded_vectorizer = pickle.load(f)
            
            # Load test data to calculate accuracy
            X_train, X_test, y_train, y_test = load_IMDB_data()
            
            # Transform the test data (FIXED: don't fit again, just transform)
            X_test_vec = loaded_vectorizer.transform(X_test)  # Only transform, don't fit
            
            # Calculate accuracy
            accuracy = loaded_classifier.score(X_test_vec, y_test)
            return JsonResponse({
                'representation': representation,
                'classifier': classifier,
                'representation_name': representation_dict.get(representation, representation),
                'classifier_name': classifier_dict.get(classifier, classifier),
                'accuracy': float(accuracy),  # Ensure it's a Python float
            })
            
        except FileNotFoundError as e:
            return JsonResponse({'error': f'Model file not found: {str(e)}'}, status=404)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'error': f'Error loading model: {str(e)}'}, status=500)
    
    else:
        return JsonResponse({'error': 'POST method required'}, status=400)

def start_active_learning(request):
    global ACTIVE_LEARNING_STATE

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            utilities = data.get("utilities", [])
            mode = data.get("mode", "simulated")  # 'simulated' or 'interactive'

            X_train, X_test, y_train, y_test = load_IMDB_data()

            vectorizer = TfidfVectorizer(max_features=10000)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            model = LogisticRegression(max_iter=1000)

            # Simulate unlabeled/labeled split
            n_initial = 100  # Initial labeled set
            np.random.seed(42)
            indices = np.random.permutation(len(X_train_vec.toarray()))

            labeled_idx = indices[:n_initial]
            unlabeled_idx = indices[n_initial:]

            X_labeled = X_train_vec[labeled_idx]
            y_labeled = np.array(y_train)[labeled_idx]

            # AL Loop (limited to 10 iterations for speed)
<<<<<<< HEAD
            if mode == "simulated":
                for i in range(10):
                    model.fit(X_labeled, y_labeled)

                    probs = model.predict_proba(X_train_vec[unlabeled_idx])
                    uncertainty = np.abs(probs[:, 1] - 0.5)
                    query_idx = np.argmin(uncertainty)

                    true_idx = unlabeled_idx[query_idx]
                    new_x = X_train_vec[true_idx]
                    new_y = y_train[true_idx]

                    X_labeled = sparse_vstack([X_labeled, new_x])
                    y_labeled = np.append(y_labeled, new_y)
                    unlabeled_idx = np.delete(unlabeled_idx, query_idx)

                    if len(y_labeled) >= 500:
                        break

                final_accuracy = model.score(X_test_vec, y_test)

                return JsonResponse({
                    "accuracy": float(final_accuracy),
                    "iterations": i + 1
                })

=======
            for i in range(10):
                model.fit(X_labeled, y_labeled)

                # Use uncertainty sampling (as example)
                probs = model.predict_proba(X_train_vec[unlabeled_idx])
                uncertainty = np.abs(probs[:, 1] - 0.5)
                query_idx = np.argmin(uncertainty)

                true_idx = unlabeled_idx[query_idx]
                new_x = X_train_vec[true_idx]
                new_y = y_train[true_idx] if mode == "simulated" else None  # Simulate label

                # Update pools
                X_labeled = sparse_vstack([X_labeled, new_x])
                y_labeled = np.append(y_labeled, new_y)
                unlabeled_idx = np.delete(unlabeled_idx, query_idx)

                # Stop if simulated mode and enough samples
                if mode == "simulated" and len(y_labeled) >= 500:
                    break

            final_accuracy = model.score(X_test_vec, y_test)
>>>>>>> 6cc5dc8665a3c92be8aee2137e603e7368e4a673

            # Save state if interactive
            if mode == "interactive":
                ACTIVE_LEARNING_STATE = {
                    "X_train_vec": X_train_vec,
<<<<<<< HEAD
                    "X_raw_train": X_train,
                    "y_train": y_train,
                    "unlabeled_idx": unlabeled_idx.tolist(),
                    "labeled_X": X_labeled,
=======
                    "y_train": y_train,
                    "unlabeled_idx": unlabeled_idx.tolist(),
                    "labeled_X": X_labeled.tolist(),
>>>>>>> 6cc5dc8665a3c92be8aee2137e603e7368e4a673
                    "labeled_y": y_labeled.tolist(),
                    "vectorizer": vectorizer,
                    "model": model,
                }

<<<<<<< HEAD
                return JsonResponse({"message": "Interactive labeling started."})
=======
            return JsonResponse({
                "accuracy": float(final_accuracy),
                "iterations": i + 1
            })
>>>>>>> 6cc5dc8665a3c92be8aee2137e603e7368e4a673

        except Exception as e:
            return JsonResponse({"error": str(e)+"start_active_learning"}, status=500)

    return JsonResponse({"error": "POST method required"}, status=400)


def submit_label(request):
    global ACTIVE_LEARNING_STATE

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            label = data.get("label")

<<<<<<< HEAD
            if not ACTIVE_LEARNING_STATE["unlabeled_idx"]:
                return JsonResponse({"error": "No more unlabeled samples."})

            idx = ACTIVE_LEARNING_STATE["unlabeled_idx"].pop(0)
            vector = ACTIVE_LEARNING_STATE["X_train_vec"][idx]

            #Correct sparse stacking
            ACTIVE_LEARNING_STATE["labeled_X"] = sparse_vstack([
                ACTIVE_LEARNING_STATE["labeled_X"], vector
            ])

            #Append label to list
            ACTIVE_LEARNING_STATE["labeled_y"].append(label)

            return JsonResponse({
                "status": "Label saved.",
                "remaining": len(ACTIVE_LEARNING_STATE["unlabeled_idx"])
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "POST method required"}, status=400)


=======
            # Get next sample
            idx = ACTIVE_LEARNING_STATE["unlabeled_idx"].pop(0)
            vector = ACTIVE_LEARNING_STATE["X_train_vec"][idx]
            
            ACTIVE_LEARNING_STATE["labeled_X"].append(vector.toarray().tolist()[0])
            ACTIVE_LEARNING_STATE["labeled_y"].append(label)

            return JsonResponse({"status": "label received", "remaining": len(ACTIVE_LEARNING_STATE["unlabeled_idx"])})
        except Exception as e:
            return JsonResponse({"error": str(e)+"submit_label"}, status=500)

    return JsonResponse({"error": "POST method required"}, status=400)

>>>>>>> 6cc5dc8665a3c92be8aee2137e603e7368e4a673
def get_next_sample(request):
    global ACTIVE_LEARNING_STATE

    if request.method == 'GET':
        try:
<<<<<<< HEAD
            if "unlabeled_idx" not in ACTIVE_LEARNING_STATE or not ACTIVE_LEARNING_STATE["unlabeled_idx"]:
                return JsonResponse({"Note": "No unlabeled data left or active learning not initialized."})

            idx = ACTIVE_LEARNING_STATE["unlabeled_idx"][0]
            review = ACTIVE_LEARNING_STATE["X_raw_train"][idx]

            return JsonResponse({"review": review})

        except Exception as e:
            return JsonResponse({"error": str(e)+get_next_sample}, status=500)

    return JsonResponse({"error": "GET method required"}, status=400)

def train_on_labeled_data(request):
    global ACTIVE_LEARNING_STATE

    if request.method == 'POST':
        try:
            if not ACTIVE_LEARNING_STATE.get("labeled_y"):
                return JsonResponse({"error": "No labeled data found."})

            X_labeled = ACTIVE_LEARNING_STATE["labeled_X"]
            y_labeled = ACTIVE_LEARNING_STATE["labeled_y"]
            model = ACTIVE_LEARNING_STATE["model"]
            vectorizer = ACTIVE_LEARNING_STATE["vectorizer"]

            # Load test set
            _, X_test, _, y_test = load_IMDB_data()
            X_test_vec = vectorizer.transform(X_test)

            # Train and evaluate
            model.fit(X_labeled, y_labeled)
            accuracy = model.score(X_test_vec, y_test)

            return JsonResponse({
                "message": "Model trained on labeled data.",
                "accuracy": float(accuracy),
                "num_samples": len(y_labeled)-100
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "POST method required."}, status=400)

=======
            idx = ACTIVE_LEARNING_STATE["unlabeled_idx"][0]
            review = ACTIVE_LEARNING_STATE["vectorizer"].inverse_transform(ACTIVE_LEARNING_STATE["X_train_vec"][idx])[0]

            return JsonResponse({"review": " ".join(review)})
        except Exception as e:
            return JsonResponse({"error": str(e)+"get_next_sample"}, status=500)

    return JsonResponse({"error": "GET method required"}, status=400)

>>>>>>> 6cc5dc8665a3c92be8aee2137e603e7368e4a673

def load_IMDB_data():
    # Load the IMDB dataset
    # This is a placeholder - you'll need to implement actual data loading
    df = pd.read_csv("./media/data/IMDB Dataset.csv")
    
    # Split features and labels
    X = df['review'].values
    y = df['sentiment'].values

    # train-test split
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)