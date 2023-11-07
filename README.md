# # INTRODUCTION:
AI-driven exploration and prediction of company registration trends with the Registrar of Companies involves using artificial intelligence and data analytics to analyse and forecast patterns in company registrations. In the rapidly evolving landscape of business and commerce, staying ahead of the curve is essential for organizations and policymakers alike.

# OBJECTIVE:
This document aims to outline the application of AI-driven exploration and prediction techniques in analysing company registration trends with the RoC. It delves into the methodologies, challenges, and 
potential benefits of using AI in this domain.

# PREDICTION:
Predicting the behavior or performance of companies is a complex task that involves analyzing various data sources and using different techniques depending on the specific aspect you want to predict.

# IMPORT THE NECESSARY LIBRARIES:
By Importing the Libraries like:
i)   Numpy
ii)  Matplotlib
iii) Seaborn
iv) Pandas

# DATA SOURCE USED:
Using  Dataset: (https://tn.data.gov.in/resource/company-master-data-tamil-nadu-upto-28th-february-2019)

# DATA COLLECTION:
To begin the exploration, we need access to a comprehensive dataset from the RoC, including company registrations, closures, industry classifications, and geographical data. This data can be obtained through official government sources and APIs.
Filepath = ‘../input/prediction-of-companies/Gov_Data.csv’
		df = pd.read_csv(Filepath)
To ensure easy visual we are creating and exploration bar graph to see the prediction of companies

# Load your dataset
data = pd.read_csv('roc.csv', encoding='latin1', low_memory=False)

# EXPLORATORY DATA ANALYSIS(EDA):
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
print(BLUE + "\nDATA CLEANING" + RESET)
missing_values = df.isnull().sum()
print(GREEN + "Missing Values : " + RESET)
print(missing_values)

# REMOVING DUPLICATE VALUE:
mean_fill = df.fillna(df.mean())
df.fillna(mean_fill, inplace=True)
duplicate_values = df.duplicated().sum()
print(GREEN + "Duplicate Values : " + RESET)
print(duplicate_values)
df.drop_duplicates(inplace=True)

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# DATA PREPROCESSING:
Steps to preprocess the data for AI modeling.
data = pd.read_csv('roc.csv', encoding='latin1', low_memory=False)

# CLEAN THE DATA:
Steps to clean the data for AI modeling.
print(BLUE + "\nDATA CLEANING" + RESET)

# DATA ANALYSIS:
print(BLUE + "\nDATA ANALYSIS" + RESET)
summary_stats = df.describe()
print(GREEN + "Summary Statistics : " + RESET)
print(summary_stats)

# MODEL ACCURACY:
print(BLUE + "\nMODELLING" + RESET);
X = df.drop("Outcome", axis=1);
y = df["Outcome"];
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 769);
scaler = StandardScaler();
X_train = scaler.fit_transform(X_train);
X_test = scaler.transform(X_test);
model = svm.SVC(kernel="linear");
model.fit(X_train, y_train);
y_pred = model.predict(X_test);
accuracy = model.score(X_test, y_test);
print(GREEN + "Model Accuracy : " + RESET);
print(accuracy);

# DATA DISTRIBUTION:
plt.figure(figsize=(10, 6))
sns.histplot(data['COMPANY_NAME'], bins=20, kde=True)
plt.title('Distribution of Registration Dates')
plt.xlabel('Registration Date')
plt.ylabel('Frequency')
plt.show()

# RESULT:
Display the sample result or performance metrices or visualization analysis obtained by using this procedure for AI-Driven Exploration and Prediction of Company Registration Trends with the Registrar of Companies (RoC)
