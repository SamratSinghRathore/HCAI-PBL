# views.py
from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from django.http import JsonResponse
import os 

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

def cold_start(request, group='experimental'):
    """
    Cold-start recommendation page with guided active learning.
    
    Args:
        request: HTTP request object.
        group: Study group ('experimental' or 'control', default 'experimental').
    
    Returns:
        Rendered cold-start template with initial movies.
    """
    # Validate group parameter
    if group not in ['experimental', 'control']:
        group = 'experimental'  # Fallback to experimental

    # Load datasets
    movies_df = pd.read_csv(os.path.join("project4", "ml-latest-small", "movies.csv"))
    ratings_df = pd.read_csv(os.path.join("project4", "ml-latest-small", "ratings.csv"))
    
    # Get popular movies
    movie_ratings = ratings_df.groupby('movieId').agg(
        rating_count=('userId', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    
    popular_movies = movie_ratings[
        (movie_ratings['rating_count'] > 100) & 
        (movie_ratings['avg_rating'] > 3.5)
    ].merge(movies_df, on='movieId')
    
    # Create diverse set of initial movies
    unique_genres = set()
    initial_movies = []
    
    for _, movie in popular_movies.iterrows():
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
    
    # Select template
    template = 'project4/cold_start.html' if group == 'experimental' else 'project4/cold_start_control.html'
    
    is_study = group in ['experimental', 'control'] and request.session.get('group') == group
    
    return render(request, template, {
        'initial_movies': json.dumps(initial_movies),
        'skip_welcome': True,
        'is_study': is_study  # Pass flag to template
    })

def submit_ratings(request):
    """Process submitted ratings and return personalized recommendations"""
    if request.method == 'POST':
        data = json.loads(request.body)
        user_ratings = data.get('ratings', {})
        
        # Convert to dataframe format
        user_df = pd.DataFrame([
            {'userId': -1, 'movieId': int(movie_id), 'rating': float(rating)}
            for movie_id, rating in user_ratings.items()
        ])
        
        # Load existing data
        movies_df = pd.read_csv(os.path.join('project4', 'ml-latest-small', 'movies.csv'))
        ratings_df = pd.read_csv(os.path.join('project4', 'ml-latest-small', 'ratings.csv'))
        
        # Combined ratings (add new user)
        combined_ratings = pd.concat([ratings_df, user_df])
        
        # Create a user-item matrix
        user_item_matrix = combined_ratings.pivot_table(
            index='userId', columns='movieId', values='rating'
        )
        
        # Fill missing values with 0
        user_item_matrix = user_item_matrix.fillna(0)
        
        # Calculate user similarity (cosine similarity)
        user_similarity = cosine_similarity(user_item_matrix)
        
        # Get similar users to our new user (userId = -1)
        new_user_idx = user_item_matrix.index.get_loc(-1)
        similar_users = [(idx, user_similarity[new_user_idx, idx]) 
                         for idx in range(len(user_item_matrix)) 
                         if idx != new_user_idx]
        
        # Sort by similarity (descending)
        similar_users.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 10 similar users
        top_similar_users = [idx for idx, _ in similar_users[:10]]
        
        # Get user IDs
        top_similar_user_ids = user_item_matrix.iloc[top_similar_users].index.tolist()
        
        # Get movies rated highly by similar users but not rated by our user
        rated_movies = set(int(movie_id) for movie_id in user_ratings.keys())
        similar_user_ratings = ratings_df[
            (ratings_df['userId'].isin(top_similar_user_ids)) & 
            (ratings_df['rating'] >= 4.0) & 
            (~ratings_df['movieId'].isin(rated_movies))
        ]
        
        # Get recommended movie count by similar users
        movie_rec_count = similar_user_ratings.groupby('movieId').size().reset_index(name='rec_count')
        
        # Merge with movie info
        recommended_movies = movie_rec_count.merge(movies_df, on='movieId')
        
        # Sort by recommendation count (descending)
        recommended_movies = recommended_movies.sort_values('rec_count', ascending=False)
        
        # Get top 10 recommendations
        top_recommendations = recommended_movies.head(10).to_dict('records')
        
        # Add explanations
        for movie in top_recommendations:
            similar_users_who_liked = similar_user_ratings[
                similar_user_ratings['movieId'] == movie['movieId']
            ]['userId'].nunique()
            movie['explanation'] = (
                f"{similar_users_who_liked} users with similar taste to yours rated this "
                f"{movie['genres'].replace('|', '/')} movie highly"
            )
        
        return JsonResponse({
            'recommendations': top_recommendations,
            'similar_user_count': len(top_similar_user_ids)
        })
    
    return JsonResponse({'error': 'Invalid request method'})

def next_questions(request):
    """Determine next best movies to rate based on current ratings"""
    if request.method == 'POST':
        data = json.loads(request.body)
        current_ratings = data.get('ratings', {})
        
        # Load datasets
        movies_df = pd.read_csv(os.path.join('project4', 'ml-latest-small', 'movies.csv'))
        ratings_df = pd.read_csv(os.path.join('project4', 'ml-latest-small', 'ratings.csv'))
        
        # Already rated movies
        rated_movies = set(int(movie_id) for movie_id in current_ratings.keys())
        
        # Extract current user genre preferences based on high ratings (≥ 4.0)
        liked_movies = [int(movie_id) for movie_id, rating in current_ratings.items() 
                       if float(rating) >= 4.0]
        
        # Get genres of liked movies
        liked_genres = set()
        if liked_movies:
            for _, movie in movies_df[movies_df['movieId'].isin(liked_movies)].iterrows():
                movie_genres = movie['genres'].split('|')
                liked_genres.update(movie_genres)
        
        # Strategies for next questions based on current state
        next_movies = []
        
        # Strategy 1: If we have some ratings, find popular movies in liked genres
        if liked_genres:
            # Get popular movies from liked genres
            genre_movies = movies_df[movies_df['genres'].str.contains('|'.join(liked_genres))]
            genre_movie_ratings = genre_movies.merge(
                ratings_df.groupby('movieId').agg(
                    rating_count=('userId', 'count'),
                    avg_rating=('rating', 'mean')
                ).reset_index(),
                on='movieId'
            )
            
            # Filter for popular, well-rated movies not yet rated
            popular_genre_movies = genre_movie_ratings[
                (genre_movie_ratings['rating_count'] > 50) &
                (genre_movie_ratings['avg_rating'] > 3.8) &
                (~genre_movie_ratings['movieId'].isin(rated_movies))
            ].sort_values('rating_count', ascending=False).head(5)
            
            for _, movie in popular_genre_movies.iterrows():
                dominant_genre = movie['genres'].split('|')[0]
                next_movies.append({
                    'id': int(movie['movieId']),
                    'title': movie['title'],
                    'genres': movie['genres'],
                    'explanation': f"Rating this will help us refine your {dominant_genre} recommendations"
                })
        
        # Strategy 2: Add movies from underrepresented genres to diversify
        current_genre_coverage = set()
        for movie_id in rated_movies:
            movie = movies_df[movies_df['movieId'] == int(movie_id)]
            if not movie.empty:
                genres = movie.iloc[0]['genres'].split('|')
                current_genre_coverage.update(genres)
        
        # Find popular movies from genres not yet covered
        popular_movies = movies_df.merge(
            ratings_df.groupby('movieId').agg(
                rating_count=('userId', 'count'),
                avg_rating=('rating', 'mean')
            ).reset_index(),
            on='movieId'
        )
        
        popular_movies = popular_movies[
            (popular_movies['rating_count'] > 100) &
            (popular_movies['avg_rating'] > 3.5) &
            (~popular_movies['movieId'].isin(rated_movies))
        ]
        
        for _, movie in popular_movies.iterrows():
            movie_genres = set(movie['genres'].split('|'))
            # Add movies with genres we haven't covered well
            if len(current_genre_coverage.intersection(movie_genres)) <= 1:
                next_movies.append({
                    'id': int(movie['movieId']),
                    'title': movie['title'], 
                    'genres': movie['genres'],
                    'explanation': f"This will help us understand if you enjoy {movie['genres'].replace('|', '/')} films"
                })
                if len(next_movies) >= 10:
                    break
        
        # If we still need more movies, add some highly rated films across all genres
        if len(next_movies) < 10:
            remaining_needed = 10 - len(next_movies)
            already_selected = set(movie['id'] for movie in next_movies)
            top_rated = popular_movies[
                (~popular_movies['movieId'].isin(already_selected)) &
                (~popular_movies['movieId'].isin(rated_movies))
            ].sort_values('avg_rating', ascending=False).head(remaining_needed)
            
            for _, movie in top_rated.iterrows():
                next_movies.append({
                    'id': int(movie['movieId']),
                    'title': movie['title'],
                    'genres': movie['genres'],
                    'explanation': f"This highly-rated movie will help us understand your general preferences"
                })
        
        return JsonResponse({'next_movies': next_movies[:10]})
    
    return JsonResponse({'error': 'Invalid request method'})

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