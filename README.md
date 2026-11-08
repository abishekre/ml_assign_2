# Email Spam Classification — End-to-End Machine Learning & Streamlit Deployment

## a. Problem Statement
Unsolicited commercial bulk email (spam) degrades organizational productivity, consumes network bandwidth, and serves as a primary delivery vector for phishing and malware. The objective of this project is to implement, evaluate, and deploy five supervised machine learning classification models capable of accurately classifying emails as **Spam (`1`)** or **Non-Spam (`0`)** based on word frequencies, special character frequencies, and capital-letter run-length statistics.

---

## b. Dataset Description
* **Dataset Name:** Spambase Dataset
* **Source:** UCI Machine Learning Repository (`https://archive.ics.uci.edu/dataset/94/spambase`)
* **Total Instances:** 4,601 rows (exceeds the minimum requirement of 500 instances)
* **Total Features:** 57 continuous numeric input features + 1 binary target column (`spam`) (exceeds the minimum requirement of 12 features)
* **Feature Breakdown:**
  * **48 Word-Frequency Features (`word_freq_make`, `word_freq_free`, `word_freq_remove`, etc.):** Continuous values in $[0, 100]$ indicating the percentage of words in the email that match a given keyword.
  * **6 Character-Frequency Features (`char_freq_;`, `char_freq_(`, `char_freq_[`, `char_freq_!`, `char_freq_$`, `char_freq_#`):** Continuous values in $[0, 100]$ indicating the percentage of characters in the email that match a specific special symbol.
  * **3 Capital Run-Length Features (`capital_run_length_average`, `capital_run_length_longest`, `capital_run_length_total`):** Numeric attributes measuring the average length, longest uninterrupted sequence, and total count of capital letters in the email.
* **Target Variable (`spam`):** Binary label where `1` indicates Spam (1,813 instances; 39.40%) and `0` indicates Non-Spam (2,788 instances; 60.60%).
* **Preprocessing Applied:**
  * Verified zero missing values across all 4,601 rows.
  * Performed an 80:20 stratified train-test split (`random_state=2026`), allocating 3,680 instances for model training and exporting 921 instances to `test_data.csv` for out-of-sample evaluation.
  * Applied `StandardScaler` within Scikit-Learn `Pipeline` objects for each classifier so that large-magnitude features (such as `capital_run_length_total`) do not dominate distance or gradient calculations, while allowing raw test CSV uploads in the Streamlit web application without retraining.

---

## c. GitHub Repository & Live Application Links
* **GitHub Repository Link:** [https://github.com/<your-github-username>/<your-repo-name>](https://github.com/<your-github-username>/<your-repo-name>)
* **Live Streamlit App Link:** [https://<your-streamlit-app-url>.streamlit.app](https://<your-streamlit-app-url>.streamlit.app)

---

## d. Models Used & Evaluation Metrics Comparison Table

All five classification models were trained on the same 3,680-instance training split and evaluated on the held-out 921-instance test set (`test_data.csv`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9164 | 0.9646 | 0.9086 | 0.8760 | 0.8920 | 0.8242 |
| Decision Tree | 0.9077 | 0.9187 | 0.8840 | 0.8815 | 0.8828 | 0.8067 |
| KNN | 0.9055 | 0.9563 | 0.8988 | 0.8567 | 0.8773 | 0.8012 |
| Naive Bayes | 0.8208 | 0.9402 | 0.7029 | 0.9449 | 0.8061 | 0.6707 |
| Support Vector Machine | 0.9262 | 0.9725 | 0.9300 | 0.8788 | 0.9037 | 0.8448 |

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Achieved the second-best overall performance (Accuracy: 0.9164, AUC: 0.9646, MCC: 0.8242). Because high-signal spam indicators (`word_freq_free`, `word_freq_remove`, `char_freq_$`, `char_freq_!`) have monotonic relationships with spam log-odds, a regularized linear boundary separated the classes well, trailing only the non-linear SVM. |
| Decision Tree | Produced balanced Precision (0.8840) and Recall (0.8815) with an F1 score of 0.8828 by creating axis-aligned threshold rules on top features like `char_freq_!` and `capital_run_length_average`. Constraining `max_depth=8` mitigated overfitting, though single-tree probability estimates yielded the lowest AUC (0.9187) among the five models. |
| KNN | Reached 0.9055 Accuracy and 0.9563 AUC using distance-weighted voting ($k=7$) after `StandardScaler` prevented capital run-length features from skewing Euclidean distances. However, because individual emails only contain a small subset of the 57 tracked keywords, high-dimensional sparsity slightly reduced Recall (0.8567). |
| Naive Bayes | Achieved the highest Recall (0.9449) across all classifiers—catching 94.5% of spam—but recorded the lowest Precision (0.7029), Accuracy (0.8208), and MCC (0.6707). Because word frequencies are zero-inflated, right-skewed, and correlated (e.g., `word_freq_hp`, `word_freq_hpl`, `word_freq_lab`), Gaussian Naive Bayes' strict independence and normality assumptions triggered excessive false positives on legitimate emails. |
| Support Vector Machine | Delivered the highest performance across five of the six metrics: Accuracy (0.9262), AUC (0.9725), Precision (0.9300), F1 (0.9037), and MCC (0.8448). The Radial Basis Function (RBF) kernel effectively captured non-linear interactions between lexical frequencies and capital-letter bursts while maintaining 93.0% precision. |
| Overall, Winner for your dataset? | **Support Vector Machine (SVM)** is the overall winner for the Spambase dataset, leading in MCC (0.8448), F1 (0.9037), AUC (0.9725), and Precision (0.9300). In production email filtering, high Precision and MCC are essential to avoid misclassifying legitimate business emails as spam, making SVM the optimal model. |

---

## Repository Structure
```text
project-folder/
├── app.py                  # Interactive Streamlit web application
├── ml_assgn_2.py           # Model training, evaluation, and serialization script
├── requirements.txt        # Python dependencies for Streamlit Cloud deployment
├── README.md               # Project documentation and evaluation comparison tables
├── test_data.csv           # Held-out 20% test dataset (921 rows, 58 columns)
└── model/                  # Saved pre-trained Scikit-Learn Pipeline models (.pkl)
    ├── Logistic_Regression.pkl
    ├── Decision_Tree.pkl
    ├── KNN.pkl
    ├── Naive_Bayes.pkl
    └── Support_Vector_Machine.pkl