# Project 1: Automated Machine Learning Interface

This project is a Django-based web application that provides a simple, end-to-end interface for supervised machine learning. It allows users to upload a dataset, visualize it, and train various classification models, demonstrating a basic automated machine learning pipeline.

## Directory Structure

```
project1/
├── static/                 # Static files (if any, e.g., CSS, JS)
├── templates/
│   └── project1/
│       ├── index.html      # Project homepage
│       └── upload.html     # Page for file upload, visualization, and training
├── urls.py                 # URL routing for the project
├── views.py                # Backend logic for data processing, plotting, and model training
├── forms.py                # Django forms for user input
└── README.md               # This file
```

## Tasks Overview

The project implements a complete, user-driven machine learning workflow, divided into the following tasks:

1.  **Task 1 & 2: Project Setup**
    -   This initial phase involved setting up the main project homepage.

2.  **Task 3: Data Loading and Visualization**
    -   **CSV Upload**: Users can upload a dataset in CSV format. The system assumes the first row contains feature names and the last column is the target variable.
    -   **Data Visualization**: After uploading, users can generate 2D scatter plots by selecting which features to plot on the X and Y axes. This helps in visually inspecting the data distribution and relationships.

3.  **Task 4: Model Training and Evaluation**
    -   **Interactive Pipeline**: Provides an interface for training a model on the uploaded data.
    -   **User Controls**: The user can configure the training process by:
        -   Selecting a machine learning model (e.g., Logistic Regression, Decision Tree, SVM).
        -   Specifying the name of the target column.
        -   Adjusting the train-test split ratio.
    -   **Evaluation**: After training, the application evaluates the model on the test set and displays key performance metrics, including Accuracy, Precision, Recall, and F1-Score.

## Pages and Buttons

The user interacts with the application through a simple, two-page workflow.

### 1. Homepage (`index.html`)

-   **URL**: `/project1/`
-   **Purpose**: Serves as the landing page for the project, providing a brief overview of its capabilities.
-   **Buttons**:
    -   **Upload a CSV File**: Navigates the user to the main interface where they can begin the machine learning workflow.

### 2. Upload & Analysis Page (`upload.html`)

-   **URL**: `/project1/upload/`
-   **Purpose**: This is the main workspace for all operations.
-   **Sections and Buttons**:
    -   **File Upload**:
        -   **Choose File**: A standard file input to select a CSV file from the local system.
        -   **Upload**: Submits the selected file to the server for processing.
    -   **Data Visualization**:
        -   **Feature Selection**: Dropdowns to select columns for the X and Y axes of a scatter plot.
        -   **Generate Plot**: Creates and displays a scatter plot based on the selected features.
    -   **Model Training**:
        -   **Configuration Form**: A form where the user can select a model, define the target column, and set the test size.
        -   **Train Model**: Submits the configuration and starts the model training and evaluation process. The results are displayed on the same page.

## Setup Instructions

1.  **Start Server**:
    ```bash
    python manage.py runserver
    ```

2.  **Access the Application**:
    -   Homepage: `http://localhost:8000/project1/`
    -   Upload & Analysis Page: `http://localhost:8000/project1/upload/`