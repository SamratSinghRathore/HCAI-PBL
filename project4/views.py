from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from django.http import JsonResponse
import os
from sklearn.decomposition import NMF

def index(request):
    """Home page view showing dataset information"""
    csv_files = [
        ('Links',   os.path.join("project4", "ml-latest-small", "links.csv")),
        ('Movies',  os.path.join("project4", "ml-latest-small", "movies.csv")),
        ('Ratings', os.path.join("project4", "ml-latest-small", "ratings.csv")),
        ('Tags',    os.path.join("project4", "ml-latest-small", "tags.csv")),
    ]

    tables = []
    for name, path in csv_files:
        try:
            # Show the first 5 rows
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

    return render(request, 'project4/index.html', {'tables': tables})

def cold_start(request, group=None):
    # Load MovieLens datasets
    movies_df = pd.read_csv(os.path.join("project4", "ml-latest-small", "movies.csv"))
    ratings_df = pd.read_csv(os.path.join("project4", "ml-latest-small", "ratings.csv"))

    # Create popular_movies DataFrame
    movie_ratings = ratings_df.groupby('movieId').agg(
        rating_count=('userId', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    
    popular_movies = movie_ratings[
        (movie_ratings['rating_count'] > 50) &  # Minimum 50 ratings
        (movie_ratings['avg_rating'] > 3.0)     # Average rating above 3
    ].merge(movies_df, on='movieId')

    # Prepare initial movies
    initial_movies = []
    unique_genres = set()
    
    for _, movie in popular_movies.sample(frac=1).iterrows():  # Shuffle for variety
        movie_genres = set(movie['genres'].split('|'))
        if len(unique_genres.intersection(movie_genres)) < 2:
            unique_genres.update(movie_genres)
            movie_data = {
                'id': int(movie['movieId']),
                'title': movie['title'],
                'genres': movie['genres'],
                'avg_rating': float(movie['avg_rating']),
                'rating_count': int(movie['rating_count']),
            }
            if group == 'experimental':
                movie_data['explanation'] = (
                    f"This {movie['genres'].replace('|', '/')} movie helps us "
                    f"understand your taste in {movie['genres'].split('|')[0]} films"
                )
            initial_movies.append(movie_data)
            if len(initial_movies) >= 10:
                break
    
    # Use different templates based on group
    if group == 'experimental':
        template = 'project4/cold_start.html'
        is_study_value = True
    elif group == 'control':
        template = 'project4/cold_start_control.html'
        is_study_value = True
    else:  # group is None (Try Movie Recommender)
        template = 'project4/cold_start.html'
        is_study_value = False
    
    return render(request, template, {
        'initial_movies': json.dumps(initial_movies),
        'skip_welcome': True,
        'is_study': is_study_value,
        'group': group
    })

def submit_ratings(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_ratings = data.get('ratings', {})
            
            # Load MovieLens data
            ratings_df = pd.read_csv(os.path.join("project4", "ml-latest-small", "ratings.csv"))
            movies_df = pd.read_csv(os.path.join("project4", "ml-latest-small", "movies.csv"))
            
            # Create user-item matrix
            user_item_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
            
            # Get movie IDs present in ratings
            rated_movie_ids = user_item_matrix.columns.tolist()
            
            # Filter movies_df to only include movies with ratings
            movies_df = movies_df[movies_df['movieId'].isin(rated_movie_ids)]
            
            # Train NMF
            model = NMF(n_components=20, init='random', random_state=42, alpha_W=0.01, alpha_H=0.01)
            U = model.fit_transform(user_item_matrix)
            V = model.components_
            
            # New user vector
            user_rated_ids = [int(mid) for mid in user_ratings.keys()]
            ratings = [float(user_ratings[mid]) for mid in user_ratings.keys()]
            
            # Map rated movie IDs to indices in V
            V_indices = [user_item_matrix.columns.get_loc(mid) for mid in user_rated_ids if mid in user_item_matrix.columns]
            V_j = V[:, V_indices]
            ratings = [r for mid, r in zip(user_rated_ids, ratings) if mid in user_item_matrix.columns]
            
            if not V_indices:
                return JsonResponse({'error': 'No valid movie IDs rated'}, status=400)
            
            # Solve for U_i
            lambda_reg = 0.01
            U_i = np.linalg.solve(V_j @ V_j.T + lambda_reg * np.eye(20), V_j @ np.array(ratings))
            
            # Predict ratings for all movies
            predictions = U_i @ V
            
            # Create movie_scores DataFrame
            movie_scores = pd.DataFrame({
                'movieId': user_item_matrix.columns,
                'score': predictions
            }).merge(movies_df[['movieId', 'title', 'genres']], on='movieId', how='left')
            
            # Exclude rated movies
            movie_scores = movie_scores[~movie_scores['movieId'].isin(user_rated_ids)]
            
            # Handle any NaN values (if merge misses some movies)
            movie_scores = movie_scores.dropna()
            
            # Get top 10 recommendations
            top_recommendations = movie_scores.sort_values(by='score', ascending=False).head(10)
            
            recommendations = [
                {'id': int(row['movieId']), 'title': row['title'], 'genres': row['genres']}
                for _, row in top_recommendations.iterrows()
            ]
            
            return JsonResponse({'recommendations': recommendations})
        except Exception as e:
            print(f"Error in submit_ratings: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def next_questions(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            rated_movies = data.get('rated_movies', [])
            
            # Load datasets
            movies_df = pd.read_csv(os.path.join("project4", "ml-latest-small", "movies.csv"))
            ratings_df = pd.read_csv(os.path.join("project4", "ml-latest-small", "ratings.csv"))
            
            # Get popular movies
            movie_ratings = ratings_df.groupby('movieId').agg(
                rating_count=('userId', 'count'),
                avg_rating=('rating', 'mean')
            ).reset_index()
            
            popular_movies = movie_ratings[
                (movie_ratings['rating_count'] > 50) & 
                (movie_ratings['avg_rating'] > 3.0)
            ].merge(movies_df, on='movieId')
            
            # Exclude rated movies
            available_movies = popular_movies[~popular_movies['movieId'].isin(rated_movies)]
            
            # Select diverse movies
            unique_genres = set()
            new_movies = []
            
            for _, movie in available_movies.sample(frac=1).iterrows():  # Shuffle for variety
                movie_genres = set(movie['genres'].split('|'))
                if len(unique_genres.intersection(movie_genres)) < 2:
                    unique_genres.update(movie_genres)
                    movie_data = {
                        'id': int(movie['movieId']),
                        'title': movie['title'],
                        'genres': movie['genres'],
                        'avg_rating': float(movie['avg_rating']),
                        'rating_count': int(movie['rating_count']),
                    }
                    if request.session.get('group') == 'experimental':
                        movie_data['explanation'] = (
                            f"This {movie['genres'].replace('|', '/')} movie helps us "
                            f"understand your taste in {movie['genres'].split('|')[0]} films"
                        )
                    new_movies.append(movie_data)
                    if len(new_movies) >= 10:
                        break
            
            return JsonResponse({'movies': new_movies})
        except Exception as e:
            print(f"Error in next_questions: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)

import random
from django.shortcuts import redirect

def start_study(request):
    """
    Initiates the user study by randomly assigning a group and redirecting to cold-start.
    """
    if not request.session.session_key:
        request.session.create()
    
    group = 'experimental' if random.random() > 0.5 else 'control'
    request.session['group'] = group
    
    try:
        return redirect('project4:cold_start_with_group', group=group)
    except Exception as e:
        print(f"Error redirecting to cold-start: {e}")
        return redirect('project4:index')

import os
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from datetime import datetime

def submit_survey(request):
    if request.method == 'POST':
        try:
            survey_data = request.POST
            group = request.session.get('group', 'unknown')
            
            data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'group': group,
                'satisfaction': survey_data.get('satisfaction', ''),
                'ease': survey_data.get('ease', ''),
                'transparency': survey_data.get('transparency', '') if group == 'experimental' else 'N/A',
                'open_feedback': survey_data.get('open_feedback', ''),
            }
            
            recommendation_data = {}
            for key in survey_data:
                if key.startswith('relevance_'):
                    movie_id = key.replace('relevance_', '')
                    recommendation_data[f'relevance_{movie_id}'] = survey_data.get(key, '')
                elif key.startswith('watch_'):
                    movie_id = key.replace('watch_', '')
                    recommendation_data[f'watch_{movie_id}'] = survey_data.get(key, '')
            
            data.update(recommendation_data)
            
            csv_path = os.path.join(settings.BASE_DIR, 'project4', 'survey_data', 'survey_responses.csv')
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            df = pd.DataFrame([data])
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, mode='w', header=True, index=False)
            
            # Clear session data
            request.session.pop('recommendations', None)
            request.session.pop('group', None)
            
            return render(request, 'project4/thank_you.html', {
                'message': 'Thank you for participating in our study!'
            })
        
        except Exception as e:
            print(f"Error processing survey: {e}")
            return render(request, 'project4/thank_you.html', {
                'message': 'An error occurred, but your participation is appreciated!'
            })
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)

from django.shortcuts import render

def study_landing(request):
    return render(request, 'project4/study_landing.html')

def survey(request):
    """
    Renders the survey page for user study participants.
    """
    # Get group from session
    group = request.session.get('group', 'unknown')
    
    # Get recommendations from session storage (passed via cold-start)
    recommendations = []
    if 'recommendations' in request.session:
        recommendations = json.loads(request.session['recommendations'])
    
    return render(request, 'project4/survey.html', {
        'recommendations': recommendations,
        'group': group
    })

def store_recommendations(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        recommendations = data.get('recommendations', [])
        request.session['recommendations'] = json.dumps(recommendations)
        return JsonResponse({'status': 'success'})
    return JsonResponse({'error': 'Invalid request method'}, status=400)