# Movie Recommender System Project

This project implements a cold-start movie recommender system using guided active learning, built with Django and the MovieLens dataset. It includes a user study to compare experimental (with explanations) and control (without explanations) interfaces.

## Prerequisites

- Python 3.11+
- Django 5.2+
- pandas
- MovieLens dataset (`ml-latest-small` directory in `project4/`)

## Installation

1. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place the MovieLens dataset (`ml-latest-small`) in `project4/`:
   ```
   project4/
   └── ml-latest-small/
       ├── movies.csv
       ├── ratings.csv
       └── ...
   ```

4. Run migrations:
   ```
   python manage.py makemigrations
   python manage.py migrate
   ```

## Running the Project

1. Start the Django development server:
   ```
   python manage.py runserver
   ```

2. Access the project at `http://127.0.0.1:8000/project4/`.

## URLs and Functionality

- **Homepage**: `/project4/`
  - Displays MovieLens dataset samples and two options:
    - **Try Movie Recommender**: Links to `/project4/cold-start/`.
    - **Go to Study**: Links to `/project4/study/`.

- **Cold-Start Recommender**: `/project4/cold-start/`
  - Rate at least 5 movies to get recommendations automatically.
  - Includes explanations (e.g., "This Comedy/Drama movie helps us understand your taste in Comedy films").

- **Study Landing Page**: `/project4/study/`
  - Offers:
    - Download a PDF with method and study details.
    - Start the user study via "Start Study" button, leading to `/project4/cold-start/experimental/` or `/project4/cold-start/control/`.

- **Study Cold-Start**:
  - `/project4/cold-start/experimental/`: Same as general recommender, with explanations.
  - `/project4/cold-start/control/`: No explanations.
  - After rating 5 movies, a "Get Recommendations and Continue" button appears, showing recommendations and leading to the survey page.

- **Survey Page**: `/project4/survey/`
  - Collects feedback on recommendations and user experience.
  - Includes a "Submit Survey" button, saving data to `project4/survey_data/survey_responses.csv`.

- **Other Endpoints**:
  - `/project4/submit-ratings/`: Processes ratings and returns recommendations (POST).
  - `/project4/next-questions/`: Loads more movies for rating (POST).
  - `/project4/store-recommendations/`: Stores recommendations for survey (POST).
  - `/project4/submit-survey/`: Saves survey responses (POST).

## Project Structure

```
project4/
├── ml-latest-small/          # MovieLens dataset
├── survey_data/              # Survey responses CSV
├── static/project4/          # Static files (e.g., study_details.pdf)
├── templates/project4/       # HTML templates
│   ├── index.html
│   ├── study_landing.html
│   ├── cold_start.html
│   ├── cold_start_control.html
│   ├── survey.html
│   ├── thank_you.html
│   └── base.html
├── urls.py                   # URL patterns
├── views.py                  # View logic
└── ...
```

## How to Use

1. **General Recommender**:
   - Visit `/project4/`, click "Try Movie Recommender".
   - Rate 5+ movies; recommendations appear automatically.

2. **User Study**:
   - Visit `/project4/`, click "Go to Study".
   - Click "Start Study" to be randomly assigned to experimental or control group.
   - Rate 5+ movies, click "Get Recommendations and Continue".
   - Complete the survey and submit feedback.

## Notes

- Ensure `project4/survey_data/` is writable for survey responses.
- The PDF (`study_details.pdf`) must be in `project4/static/project4/`.
- For production, run `python manage.py collectstatic` and configure a WSGI server.

## Troubleshooting

- **URL Errors**: Verify `urls.py` includes all patterns listed above.
- **Template Errors**: Ensure all templates are in `project4/templates/project4/`.
- **Missing Data**: Confirm `ml-latest-small` is in `project4/`.
- **Survey Issues**: Check `survey_data/survey_responses.csv` for saved responses.