# ModelBench-Pro 🚀

**An Interactive Machine Learning Experiment Tracking and Evaluation Platform**

ModelBench-Pro is a Python-based machine learning platform that helps users upload datasets, preprocess data, train classification models, evaluate performance, compare algorithms, run cross-validation, and track experiments through an interactive Streamlit dashboard.

## ✨ Features

- **Dataset Upload:** Upload CSV datasets for machine learning experiments.
- **Data Validation:** Check datasets for empty data, duplicate columns, and insufficient rows or columns.
- **Data Preprocessing:** Handle missing values, scale numerical features, and encode categorical features.
- **Model Training:** Train multiple classification algorithms.
- **Model Evaluation:** Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- **Model Comparison:** Compare classification models through visualizations.
- **Cross-Validation:** Evaluate model performance across multiple folds.
- **Experiment Tracking:** Save and review experiment history.
- **Model Download:** Download trained models for later use.
- **Interactive Dashboard:** Access the workflow through a Streamlit web interface.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web dashboard |
| Pandas | Dataset manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, training, and evaluation |
| Matplotlib | Data visualization |
| Joblib | Model persistence |
| Pytest | Automated testing |

## 🤖 Supported Models

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Classifier (SVC)

## 📁 Project Structure

```text
ModelBench-Pro/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── config/
│   └── settings.py
├── data/
│   ├── README.md
│   └── sample/
├── src/
│   └── modelbench/
│       ├── data/
│       │   ├── loader.py
│       │   ├── validator.py
│       │   ├── cleaner.py
│       │   └── preprocessing.py
│       ├── models/
│       │   ├── registry.py
│       │   ├── trainer.py
│       │   └── predictor.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── cross_validation.py
│       │   └── visualizations.py
│       ├── experiments/
│       │   ├── tracker.py
│       │   └── repository.py
│       └── utils/
│           ├── logging_config.py
│           └── file_utils.py
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   └── test_experiments.py
├── database/
├── models/
├── reports/
└── screenshots/
```

## ⚙️ Installation

### Prerequisites

- Python installed on your system
- Git
- pip

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ModelBench-Pro
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Python path

```bash
export PYTHONPATH="$PWD/src"
```

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## 🚀 How to Use

1. Launch the Streamlit application.
2. Upload a CSV dataset.
3. Inspect and validate the dataset.
4. Select the target column you want to predict.
5. Choose a classification algorithm.
6. Train the model and review evaluation metrics.
7. Compare models or run cross-validation.
8. Save and review experiments in the experiment history.
9. Download the trained model if needed.

**Note:** ModelBench-Pro is designed for classification tasks. Use a dataset with a suitable categorical target column.

## 🧪 Run Tests

The project uses pytest for automated testing.

```bash
source venv/bin/activate
export PYTHONPATH="$PWD/src"
pytest -v
```

### Test Results

The current reported test run passed all 25 tests.

- **Tests passed:** 25
- **Tests failed:** 0
- **Execution time:** 2.21 seconds

Run the tests again after making code changes to verify the latest version.

## 📊 Evaluation Metrics

ModelBench-Pro calculates:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Confusion matrix
- Classification report

These metrics help assess and compare the performance of trained classification models.

## 🔮 Future Improvements

Potential future enhancements include:

- Support for regression algorithms
- Hyperparameter tuning
- More dataset visualizations and exploratory data analysis
- Experiment export to CSV or JSON
- Improved experiment filtering and search
- Additional model evaluation reports
- Deployment to a public hosting platform

## 🎯 Project Objective

The goal of ModelBench-Pro is to simplify machine learning experimentation by bringing dataset preparation, model training, evaluation, comparison, and experiment tracking into one interface.

## 👨‍💻 Author

**Anup Kumar**

- GitHub: Add your GitHub profile URL
- LinkedIn: Add your LinkedIn profile URL

## 📄 License

This project is distributed under the license included in the repository. See the `LICENSE` file for details.
