# 🧠 HCAI ASSIGNMENT

## 👨‍💻 DEVELOPERS:

| Developer Name        | GITHUB username   | Matriculation Number                     |
| --------------- | --------------- | ------------------------------- |
| **Ankit Rathore**            | **SamratSinghRathore**         | 641313                       |
| **Surya Pratap Singh Rathor**      | **spsrathor**    | 641312 |

A Django-based web application that allows users to:

- **Upload CSV files**
- **Visualize datasets**
- **Train machine learning models (Logistic Regression, Decision Tree, SVM)**

---

## 📁 Features

### 1. **Upload CSV**
Upload a `.csv` file through a form. The file is temporarily stored in the server under `MEDIA_ROOT/uploads`.

### 2. **Data Visualization**
If the uploaded dataset contains Iris-like features (e.g., `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `species`), the app generates:
- A **scatter plot** of sepal and petal dimensions grouped by species.  
Otherwise, it shows **countplots** of top object-type columns.

### 3. **Train ML Models**
You can train one of three models:
- **Logistic Regression**
- **Decision Tree**
- **Support Vector Machine (SVM)**

Each model supports basic hyperparameter tuning through a form. The app splits the dataset, trains the selected model, and displays:
- The **scoring metric** (accuracy, precision, recall, or F1)
- The model used
- The score result

---

## 🔁 Flow Overview

1. **Home Page (`/`)**
   - Basic landing page.

2. **Upload Page (`/upload/`)**
   - Upload a CSV file.
   - Choose whether to proceed to:
     - Data Plotting (`/plot/`)
     - Model Training (`/train_model/`)

3. **Plot Page (`/plot/`)**
   - Visualizes the dataset using Matplotlib and Seaborn.
   - Displays either scatter plots or countplots based on data type.

4. **Model Training (`/train_model/`)**
   - Choose target column, test size, model type, and metric.
   - Displays model performance in a formatted result view.

---

## 🔧 Installation & Setup

1. **Clone the repo**
```bash
git clone https://github.com/SamratSinghRathore/HCAI-PBL


# Install dependencies: 

https://github.com/SamratSinghRathore/HCAI-PBL


# Start development server

python manage.py runserver

```

## 📌 Routes

| URL Path        | View Function   | Description                     |
| --------------- | --------------- | ------------------------------- |
| `/`             | `index`         | Home page                       |
| `/upload/`      | `upload_csv`    | Upload CSV and choose next step |
| `/plot/`        | `generate_plot` | Visualize uploaded dataset      |
| `/train_model/` | `train_model`   | Train and evaluate ML model     |
