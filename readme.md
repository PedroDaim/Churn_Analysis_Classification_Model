**Churn Analysis Classification Model**

**Project Overview**

This project aims to develop a robust classification model to predict whether a customer will churn (a binary outcome: churn/no churn). We will follow a standard machine learning workflow, from initial data acquisition and exploration to model development, improvement, and preparation for deployment.
The primary goal is to leverage the Telco Churn Dataset to build a predictive model that can provide insights into factors influencing churn and enable proactive customer retention strategies.

**Project Phases**

The project is structured into five distinct phases, ensuring a systematic approach to model development and deployment readiness.

**Phase 1: Data Acquisition and Initial Exploration**

**Understand the Tour & Travels Churn Dataset:** Load the dataset and gain a comprehensive understanding of its features (columns) and the target variable (the binary churn indicator).

**Dataset Source:** https://www.kaggle.com/datasets/tejashvi14/tour-travels-customer-churn-prediction/data

**Initial Data Inspection:** Perform preliminary checks to identify missing values, verify data types, and compute basic descriptive statistics for all features.

****Exploratory Data Analysis (EDA):**** Visualize relationships between various features and the target variable. This includes identifying potential correlations, distributions, and patterns that might indicate drivers of churn.

**Phase 2: Data Preprocessing and Feature Engineering**

This phase prepares the raw data for machine learning models.

**Handling Missing Values:** Implement a strategy to address any missing data points. This may involve imputation techniques (e.g., mean, median, mode, or more sophisticated methods) or judicious removal of records/features.

**Encoding Categorical Variables:** Convert non-numerical categorical features (e.g., 'Income',) into a numerical format that machine learning algorithms can process. Common techniques include One-Hot Encoding or Label Encoding.

**Feature Scaling:** Standardize or normalize numerical features to ensure that all features contribute equally to the model, preventing features with larger scales from dominating the learning process.

**Feature Engineering:** Create new, more informative features from existing ones. Examples include calculating total services subscribed, tenure in months, or derived ratios that could enhance predictive power.

**Phase 3: Model Development**

This phase involves building and evaluating the predictive model.

**Splitting Data:** Divide the prepared dataset into training and testing sets. The training set will be used to train the selected machine learning model, while the testing set will serve as unseen data to evaluate its generalization performance.

**Model Selection:** Choose an appropriate classification algorithm from the scikit-learn library. Initial candidates include simpler models like Logistic Regression, Decision Tree Classifier, or RandomForest Classifier. For more advanced challenges, Gradient Boosting Classifiers such as XGBoost or LightGBM can be explored.

**Model Training:** Train the chosen classification model using the training dataset.

**Model Evaluation:** Assess the trained model's performance on the testing set using appropriate classification metrics. Key metrics include:

**Accuracy:** The proportion of correctly classified instances.

**Precision:** The proportion of positive identifications that were actually correct.

**Recall (Sensitivity):** The proportion of actual positives that were identified correctly.

**F1-Score:** The harmonic mean of precision and recall.

**ROC AUC Score:** Measures the area under the Receiver Operating Characteristic curve, indicating the model's ability to distinguish between classes.

**Confusion Matrix:** A table that summarizes the performance of a classification algorithm.

**Phase 4: Model Improvement and Interpretation (Iterative Process)**

This iterative phase focuses on optimizing the model and understanding its predictions.

**Hyperparameter Tuning:** Optimize the model's performance by systematically adjusting its hyperparameters. Techniques like Grid Search or Random Search Cross-Validation can be employed.

**Feature Importance (for some models):** For models that support it (e.g., tree-based models), analyze which features contribute most significantly to the model's predictions. This provides valuable business insights.

**Refinement:** Based on the evaluation results and feature importance insights, iterate on previous phases. This might involve revisiting preprocessing steps, engineering new features, or experimenting with different model architectures.

**Phase 5: Deployment Preparation (for Streamlit App)**

This phase prepares the trained model for potential integration into a web application.

**Model Saving:** Save the trained classification model using a format like pickle or joblib so it can be loaded and used later by a Streamlit application without retraining.

**Preprocessing Pipeline:** Create a scikit-learn pipeline that encapsulates all preprocessing steps (e.g., imputation, encoding, scaling) along with the trained model. This ensures that new, unseen data is transformed consistently before making predictions, mimicking the training environment.

**Installation**

To set up the project locally, follow these steps:

Clone the repository:

git clone https://github.com/PedroDaim/Churn_Analysis_Classification_Model
cd churn-analysis-classification-model

Create a virtual environment (recommended):

python -m venv venv

Activate the virtual environment:

On Windows:

.\venv\Scripts\activate

On macOS/Linux:

source venv/bin/activate

Install the required dependencies:

pip install -r requirements.txt

Usage
Once the dependencies are installed, you can run the project scripts. Specific instructions will be provided as the project progresses and scripts are developed for each phase.

Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests.

License
This project is licensed under the MIT License. See the LICENSE file for details.
