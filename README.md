# Cancer-Risk-Prediction-Model
This project builds a machine learning model to predict cancer risk based on various features such as age, gender, smoking habits, BMI, blood pressure, cholesterol, and genetic factors. The model uses the RandomForestClassifier algorithm to predict whether a person has cancer risk (1) or not (0).

# Libraries Used
The following Python libraries are required for the project:
numpy
pandas
matplotlib
seaborn
scikit-learn

To install these libraries, run the following command in your terminal or command prompt:
pip install numpy pandas matplotlib seaborn scikit-learn

# Dataset
The dataset used in this project contains the following columns:

Age: Age of the individual
Gender: Gender (Female = 0, Male = 1)
Smoking: Smoking habits (0: Non-smoker, 1: Smoker)
BMI: Body Mass Index
BloodPressure: Blood pressure level
Cholesterol: Cholesterol level
GeneticFactor: Genetic factor (0: No, 1: Yes)
CancerRisk: Cancer risk (0: No risk, 1: Risk)

The Classification Report and Confusion Matrix will be displayed, showing the model's accuracy and performance.
The model will predict cancer risk for the test dataset and classify individuals as either having a cancer risk or not.
Feature Importance: After training, a bar chart is generated showing the importance of each feature in predicting cancer risk.

Project Results
The model outputs:

Classification Report: Metrics such as accuracy, precision, recall, and f1-score.
Confusion Matrix: Shows the true positive, false positive, true negative, and false negative results.
Feature Importance: A bar graph displaying which features have the most influence on the model’s predictions.
Development Notes
Data Cleaning: Ensure any missing or inconsistent data is handled appropriately.
Model Selection: This project uses RandomForestClassifier, but you could try other machine learning algorithms for comparison.
Hyperparameter Tuning: Experiment with hyperparameters to improve the model’s performance.
