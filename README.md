
# Fake Job Posting Detection

This project focuses on detecting fraudulent job postings using machine learning. It applies multiple classification algorithms on a labeled dataset and compares their performance to identify which model performs best in distinguishing real and fake job advertisements.

## Dataset

- **Source**: `fake job posting.xlsx`
- **Target Variable**: `fraudulent` (binary classification)

The dataset contains features like job title, location, company profile, description, requirements, etc.

---

## Features

- Data cleaning and preprocessing using custom text normalization
- TF-IDF vectorization on combined text fields
- Model training using:
  - Decision Tree
  - Random Forest
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - K-Nearest Neighbors (KNN)
  - Neural Networks (MLP Classifier)
- Comparison of accuracy scores
- Exported detailed performance metrics (accuracy, confusion matrix, classification report) to a Markdown file
- Final accuracy plot visualization

---

## Installation & Requirements

Install the following Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn seaborn nltk openpyxl
```

Make sure to download the NLTK stemmer:
```python
import nltk
```

---

## Data Preprocessing

- Text cleaning:
  - Lowercasing
  - Removal of emails, URLs, HTML tags, punctuations
  - Stop word removal
  - Stemming using `PorterStemmer`
- Null value handling with mode imputation
- TF-IDF vectorization on combined text fields with `max_features=100`

---

## Models Used

Each model is trained and evaluated on the same split:

- Train/Test Split: `80/20`
- Evaluation Metrics:
  - Accuracy
  - Confusion Matrix
  - Classification Report

The results are saved in `Performance Result.md`.

---

## Output Example

A line graph is plotted to compare the accuracy of all models on the test dataset:

```python
models=["Decision Tree","Random Forest","Logistic Regression","SVC","KNN","Neural Networks"]
```

The chart is rendered using `matplotlib`.

---

## File Structure

```bash
.
├── Detect_Fake.py               # Main script with data 
├── fake job posting.xlsx        # Input dataset
├── Performance Result.md        # Output file containing accuracy, confusion matrix & classification reports
```

---

## Sample Plot

A plot is displayed at the end showing accuracy comparison among all models on the test set.

---

## Future Improvements

- Use advanced NLP models like BERT or Word2Vec
- Add GUI interface using Tkinter
- Add model explainability (SHAP/LIME)

---

## Author

Developed by **Syed Areeb Ashraf**
