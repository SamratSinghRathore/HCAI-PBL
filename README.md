# 🎓 Active Learning for Text Classification (IMDB Dataset)

This project implements Active Learning techniques for sentiment analysis on the IMDB 50K movie reviews dataset. It is designed to demonstrate supervised learning (baseline) and various active learning strategies, including simulated and interactive labeling interfaces.

---

## 📂 Project Overview

The Django application is structured around **three core tasks**:

### ✅ Task 1: Supervised Learning (Baseline)
Train a model on the entire labeled training dataset using:
- Text representations: **TF-IDF** or **Bag of Words (BoW)**
- Classifiers: **Logistic Regression** or **Neural Network**

### ✅ Task 2: Pool-Based Active Learning
Train a model using **Active Learning** strategies from a small labeled pool:
- Utility functions:
  - Uncertainty Sampling
  - Diversity Sampling
  - Query-by-Committee
  - Expected Error Reduction
- Modes:
  - Simulated user (auto-label)
  - Interactive user (manual labeling)

### ✅ Task 3: Interactive Labeling
Label reviews manually via a user interface:
- Dynamically presents reviews
- User selects positive or negative
- Tracks progress

---

## 🚀 Getting Started

### 🔧 Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourname/active-learning-imdb.git
   cd active-learning-imdb
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations**:
   ```bash
   python manage.py migrate
   ```

5. **Run the server**:
   ```bash
   python manage.py runserver
   ```

6. **Place the dataset**:
   Download and place the [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) CSV file at:
   ```
   ./media/data/IMDB Dataset.csv
   ```

---

## 🧠 How to Use the Interfaces

Open your browser and go to:  
[http://127.0.0.1:8000/project2/](http://127.0.0.1:8000/project2/)

---

### 🔹 Task 1: Train Full Model
1. Select **Text Representation** (TF-IDF or BoW)
2. Select **Classifier** (LogReg or NeuralNet)
3. Click **Train Full Model**
4. Or click **Load Pre-trained Model** to load from disk
5. View accuracy and training time

---

### 🔹 Task 2: Simulated Active Learning
1. Select utility function(s) (e.g., Uncertainty Sampling)
2. Choose **Simulated User**
3. Click **Start Active Learning**
4. Wait for results: accuracy and number of iterations are shown

---

### 🔹 Task 3: Interactive Labeling
1. Select utility function(s)
2. Choose **Interactive Labeling Interface**
3. Click **Start Active Learning**
4. A review appears in the textbox
5. Click **Positive** or **Negative** to label
6. Progress is shown as you label reviews
7. (Optional) Train the model later on labeled data

---

## 📁 File Structure Highlights

```bash
project2/
├── views.py                # Django views: training, active learning, labeling
├── urls.py                 # URL routes
├── templates/project2/     # HTML templates (index.html, upload.html, etc.)
├── models/                 # Saved models and vectorizers
│   ├── model_tfidf_logreg.pkl
│   ├── vectorizer_tfidf.pkl
│   └── current_model_metadata.json
├── forms.py                # (optional) for file uploads
├── static/                 # (optional) for custom JS/CSS
```

---

## 💡 Extensions You Can Add

- ✅ Batch Active Learning
- ✅ Stream-based Active Learning
- 📈 Accuracy curves or learning plots
- 💾 Export labeled data
- 🧠 Add SVM, BERT, or Word2Vec encoders

---


## Special Note for Task2 and Task3:

---

## 🧪 Using Active Learning Modes (Simulated vs Interactive)

### ▶️ Simulated Mode (Automatic Labeling)

Use this to simulate labeling using the existing IMDB dataset labels.

#### ✅ Steps:
1. Under **Task 2: Pool-based Active Learning**, check one or more **Utility Functions**:
   - Uncertainty Sampling
   - Diversity Sampling
   - Query-by-Committee
   - Expected Error Reduction
2. Select **"Simulated User (Automatic)"**.
3. Click **"Start Active Learning"**.
4. Wait for the results:
   - Final Accuracy
   - Number of Iterations
   - Utility Strategy used

🧠 The backend will simulate an active learning loop by auto-labeling the most informative examples using ground-truth labels.

---

### 👤 Interactive Mode (Manual Labeling)

Use this mode to label reviews manually as if you're the oracle/human-in-the-loop.

#### ✅ Steps:
1. Under **Task 2: Pool-based Active Learning**, check your desired **Utility Function(s)**.
2. Select **"Interactive Labeling Interface"**.
3. Click **"Start Active Learning"**.
4. A new section will appear:
   - A review will appear in a read-only textbox.
   - Below it, click **"Positive"** or **"Negative"** to label the review.
5. Your labeling progress is tracked.
6. After labeling N samples, you can (optionally) train a model on the labeled data.

⚠️ Make sure to **click "Start Active Learning"** after selecting "Interactive" — this initializes the backend state and prepares the review pool.

---

### ⏹️ Stopping Criteria / Completion

- In **Simulated Mode**, the system automatically stops after a fixed number of iterations (e.g., 10 or 500 labeled samples).
- In **Interactive Mode**, you can stop anytime. After labelling, you can use the **Train on Labeled Data** button to evaluate the model on your labeled data.



## 🧑‍💻 Credits

- Built using **Django** and **Scikit-learn**
- Dataset: [IMDB 50K Reviews from Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 📃 License

MIT License – Feel free to use and adapt!



