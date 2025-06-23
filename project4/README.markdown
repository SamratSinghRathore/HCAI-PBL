# Project 4: Influence of Future Predictions Over Active Learning of Users' Tastes for Recommender Systems

This project implements a movie recommender system using the MovieLens dataset, focusing on a guided active learning setup for cold-start users, a user study to test the impact of providing explanations, and a user interface for the study. The system is built using Django, with templates for user interaction and views to handle backend logic. The MovieLens small dataset (100k ratings, 9k movies, 600 users) is used for development.

## Directory Structure

```
project4/
├── ml-latest-small/        # MovieLens dataset files (links.csv, movies.csv, ratings.csv, tags.csv)
├── survey_data/            # Stores survey_responses.csv for user study feedback
├── static/                 # Static files (e.g., JS, CSS)
│   └── project4/
│       └── js/
│           └── cold_start.js  # JavaScript for rating submission and navigation
├── templates/
│   └── project4/
│       ├── cold_start.html        # Experimental group cold-start interface
│       ├── cold_start_control.html # Control group cold-start interface
│       ├── index.html             # Homepage
│       ├── study_landing.html     # User study landing page
│       ├── survey.html            # Feedback survey page
│       └── thank_you.html         # Thank you page after survey submission
├── urls.py                 # URL routing for the project
├── views.py                # Backend logic for rendering pages and processing data
└── README.md               # This file
```

## Tasks Overview

The project addresses three main tasks:

1. **Task 1: Guided Active Learning Setup**
   - Implemented a cold-start recommendation system where users rate movies to receive personalized recommendations.
   - The experimental group receives explanations (e.g., "This movie helps us understand your taste in [genre] films") to guide their ratings, while the control group does not.
   - Uses cosine similarity for recommendations based on user ratings.

2. **Task 2: User Study Design**
   - Hypothesis: Providing explanations improves recommendation accuracy and user satisfaction.
   - Design: A between-subjects experiment with 100 participants (50 experimental, 50 control), recruited via platforms like Prolific.
   - Procedure: Users rate movies, receive recommendations, and complete a survey.
   - Metrics: Mean Absolute Error (MAE) for accuracy, user satisfaction, ease of use, and perceived transparency (experimental group only).

3. **Task 3: User Study Interface**
   - A landing page (`study_landing.html`) provides access to a PDF with study details and a button to start the study.
   - Interfaces for rating movies (`cold_start.html`, `cold_start_control.html`) and providing feedback (`survey.html`).
   - Recommendations are generated after rating at least 5 movies, followed by a survey to collect feedback.

## Pages and Buttons

Below is a description of each page and its interactive elements, aligned with the project tasks.

### 1. Homepage (`index.html`)
- **URL**: `/project4/`
- **Purpose**: Introduces the recommender system and provides access to the general recommender and user study.
- **Content**:
  - Displays sample data from the MovieLens dataset (first 5 rows of `links.csv`, `movies.csv`, `ratings.csv`, `tags.csv`).
  - Links to try the recommender or join the user study.
- **Buttons**:
  - **Try Movie Recommender** (`<a href="{% url 'project4:cold_start' %}" class="btn btn-primary">Try Movie Recommender</a>`):
    - Redirects to `/project4/cold-start/` for general use of the recommender (experimental mode with explanations).
    - Users can rate movies and receive recommendations without participating in the study.
  - **Go to Study** (`<a href="{% url 'project4:study_landing' %}" class="btn btn-outline-primary">Go to Study</a>`):
    - Redirects to `/project4/study/` to access the user study landing page.

### 2. User Study Landing Page (`study_landing.html`)
- **URL**: `/project4/study/`
- **Purpose**: Entry point for the user study, providing information and initiating participation (Task 3).
- **Content**:
  - Welcomes users to the study and explains its goal (improving movie recommendations).
  - Offers a downloadable PDF with details about the recommendation method (Task 1) and study design (Task 2).
- **Buttons**:
  - **Download PDF** (`<a href="{% static 'project4/study_details.pdf' %}" class="btn btn-outline-primary" download>Download PDF</a>`):
    - Downloads `study_details.pdf` (not implemented in provided code; assumed to be in `static/project4/`).
    - Contains explanations of the guided active learning method and user study design.
  - **Start Study** (`<a href="{% url 'project4:start_study' %}" class="btn btn-primary">Start Study</a>`):
    - Redirects to `/project4/start-study/`, which randomly assigns the user to the experimental or control group and redirects to `/project4/cold-start/<group>/`.

### 3. Cold-Start Interface (Experimental: `cold_start.html`, Control: `cold_start_control.html`)
- **URLs**:
  - General: `/project4/cold-start/` (experimental mode, no study context).
  - Study: `/project4/cold-start/experimental/` or `/project4/cold-start/control/`.
- **Purpose**: Collects user ratings for movies to generate personalized recommendations (Task 1).
- **Content**:
  - Displays a list of popular movies (initially 10) with genres and average ratings.
  - Experimental group (`cold_start.html`): Includes explanations (e.g., "This [genre] movie helps us understand your taste in [genre] films").
  - Control group (`cold_start_control.html`): No explanations, only movie details.
  - Users must rate at least 5 movies to proceed.
- **Buttons and Interactions**:
  - **Star Rating System** (`<span class="star-rating">★</span>`):
    - Users click stars to rate movies from 0.5 to 5.0.
    - Updates dynamically to show "Your rating: [value]" or "Not rated".
  - **Load More Movies** (`<button id="load-more" class="btn btn-outline-secondary">Load More Movies</button>`):
    - Calls `/project4/next-questions/` to fetch additional movies based on current ratings.
    - Prioritizes movies from liked genres or underrepresented genres to diversify ratings.
  - **Get Recommendations and Continue** (`<button id="submit-ratings" class="btn btn-primary">Get Recommendations and Continue</button>`):
    - Enabled after rating 5+ movies.
    - In study mode (`is_study=true`): Submits ratings to `/project4/submit-ratings/`, stores recommendations via `/project4/store-recommendations/`, and redirects to `/project4/survey/`.
    - In general mode: Fetches recommendations and displays them on the same page.

### 4. Survey Page (`survey.html`)
- **URL**: `/project4/survey/`
- **Purpose**: Collects user feedback on recommendations and the rating process (Tasks 2 and 3).
- **Content**:
  - Lists recommended movies with genres and explanations (e.g., "[N] users with similar taste rated this [genre] movie highly").
  - Questions per recommendation:
    - Relevance (Not relevant to Very relevant).
    - Willingness to watch (Yes/No).
  - General questions:
    - Satisfaction with recommendations (Not satisfied to Very satisfied).
    - Ease of use of the rating interface (Very difficult to Very easy).
    - Experimental group only: Transparency of the system (Not transparent to Very transparent).
  - Open-ended feedback textarea.
- **Buttons**:
  - **Submit Survey** (`<button type="submit" class="btn btn-primary">Submit Survey</button>`):
    - Submits form data to `/project4/submit-survey/`.
    - Saves responses to `survey_data/survey_responses.csv`.
    - Redirects to `/project4/thank_you.html`.

### 5. Thank You Page (`thank_you.html`)
- **URL**: Rendered after survey submission.
- **Purpose**: Confirms participation and provides closure.
- **Content**:
  - Displays a message (e.g., "Thank you for participating in our study!" or an error message if submission fails).
- **Buttons**:
  - **Return to Home** (`<a href="{% url 'project4:index' %}" class="btn btn-primary">Return to Home</a>`):
    - Redirects to `/project4/` (homepage).

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install django pandas numpy scikit-learn
   ```

2. **Place MovieLens Dataset**:
   - Download the MovieLens small dataset (`ml-latest-small.zip`) from the [MovieLens website](https://grouplens.org/datasets/movielens/).
   - Extract `links.csv`, `movies.csv`, `ratings.csv`, and `tags.csv` to `project4/ml-latest-small/`.

3. **Configure Django**:
   - Add `project4` to `INSTALLED_APPS` in `settings.py`.
   - Ensure middleware includes:
     ```python
     MIDDLEWARE = [
         ...
         'django.middleware.csrf.CsrfViewMiddleware',
         'django.contrib.sessions.middleware.SessionMiddleware',
         ...
     ]
     ```
   - Set up static files:
     ```python
     STATIC_URL = '/static/'
     STATICFILES_DIRS = [BASE_DIR / "project4/static"]
     ```

5. **Start Server**:
   ```bash
   python manage.py runserver
   ```

6. **Access the Application**:
   - Homepage: `http://localhost:8000/project4/`
   - Study: `http://localhost:8000/project4/study/`

