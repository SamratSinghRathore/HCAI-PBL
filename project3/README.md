# Project 3: Explainable AI (XAI) with Palmer Penguins

This project explores machine learning explainability using the Palmer Penguins dataset. It provides a hands-on interface to demonstrate two key XAI concepts: the trade-off between model complexity and interpretability, and the generation of counterfactual explanations.

## Directory Structure

```
project3/
├── static/                 # Static files (e.g., generated plots)
│   └── project3/
├── templates/
│   └── project3/
│       ├── index.html             # Project homepage with navigation
│       ├── decision_tree.html     # Interface for Decision Tree analysis
│       ├── logistic_regression.html # Interface for Logistic Regression analysis
│       └── counterfactual.html    # Interface for Counterfactual Explanations
├── urls.py                 # URL routing for the project
├── views.py                # Backend logic for all XAI tasks
└── README.md               # This file
```

## Tasks Overview

The project is divided into two main parts, covering four tasks:

1.  **Interpretability and Model Complexity (Tasks 1-3)**
    -   **Task 1 & 2: Decision Tree Interpretability**: Implements an interface where a user can train a Decision Tree classifier. A slider controls the `ccp_alpha` parameter for cost-complexity pruning, allowing the user to directly observe the trade-off between model sparsity (fewer leaves) and accuracy.
    -   **Task 3: Logistic Regression Interpretability**: Provides an interface for training a Logistic Regression model where the user can select which features to include. This demonstrates how the number of features (a measure of complexity) impacts model performance and feature importance.

2.  **Counterfactual Explanations (Task 4)**
    -   Implements an interface to generate counterfactuals. The system selects a data point, predicts its class, and then calculates the minimal feature changes required to flip the prediction to a different, desired class.

## Pages and Buttons

The application is organized into a main index page and three distinct analysis pages.

### 1. Homepage (`index.html`)

-   **URL**: `/project3/`
-   **Purpose**: Acts as a central hub for the project.
-   **Buttons**:
    -   **Generate Decision Tree Analysis →**: Navigates to the Decision Tree interpretability interface.
    -   **Generate Logistic Regression Analysis →**: Navigates to the Logistic Regression feature selection interface.
    -   **Generate Counterfactual Analysis →**: Navigates to the counterfactual explanations interface.

### 2. Decision Tree Analysis Page (`decision_tree.html`)

-   **URL**: `/project3/decision-tree/`
-   **Purpose**: To visualize the effect of sparsity on a decision tree.
-   **Interactions**:
    -   **Sparsity Parameter (λ) Slider**: Allows the user to adjust the cost-complexity pruning parameter. The page reloads automatically to show the updated model.
    -   **Outputs**: Displays the pruned decision tree as an image, along with its test accuracy and the current number of leaves.

### 3. Logistic Regression Analysis Page (`logistic_regression.html`)

-   **URL**: `/project3/logistic-regression/`
-   **Purpose**: To explore the impact of feature selection on a linear model.
-   **Interactions**:
    -   **Feature Selection Checkboxes**: The user can check or uncheck features to include in the model.
    -   **Retrain Model**: Submits the selected features to train a new model.
    -   **Outputs**: Displays the model's accuracy and a bar chart showing the importance (absolute coefficient value) of the selected features.

### 4. Counterfactual Explanations Page (`counterfactual.html`)

-   **URL**: `/project3/counterfactual/`
-   **Purpose**: To show users "what-if" scenarios for a model's prediction.
-   **Interactions**:
    -   **Find New Counterfactual**: Reloads the page to select a new random data point and generate a new explanation.
    -   **Outputs**:
        -   Displays the features of a randomly selected "query instance" and its predicted species.
        -   Shows a table of counterfactuals, detailing the minimal feature changes needed to alter the prediction to a desired