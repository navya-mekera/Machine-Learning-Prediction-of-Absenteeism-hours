# Machine-Learning-Prediction-of-Absenteeism-hours

### Introduction
This project is a study on predicting absenteeism at work. The objective is - given a set of data points that depicts the frequency of absence due to various health reasons, in a given time period, coupled with absentee’s personal information such as commute distance, age, children etc., we want to be able to predict the number of hours the employee would be absent for using machine learning algorithms. Having the insight and the ability to predict future absenteeism is useful to developing companies, in the sense that they can make necessary adjustments to changing circumstances in advance, which gives it a competitive advantage in the market.


#### Goal
The goal of the project is to evaluate which machine learning algorithms can best classify and predict Absenteeism category in test data. To do so, we first evaluated each model using the train dataset and ranked the models and reapplied them to the test data.


### Data
The description of the data can be found [here](https://github.com/navyamh24/Machine-Learning-Prediction-of-Absenteeism-hours/blob/master/Data%20Sets/DatasetDescription.md), also in the Data folder


### Overview of Methodology:

The analysis follows this 5-step approach-
1.	Data Cleaning and Preprocessing: Clean and preprocess the data set to make it analysis ready 
2.	Data Exploration: Exploring patterns in the data through data visualizations
3.	Model Building: Build Machine learning models using the training data set
4.	Model Evaluation: Evaluate the performance of each of the models under consideration on validation data set
5.	Testing: Test the models evaluated as best ones, on the test data for final prediction accuracy on the test dataset

Each of these steps have been explained in further detail in their respective sections python notebooks.

#### Models used
As part of model building, various Machine learning techniques incorporated are- Decision Tree, Naïve Bayes, Support Vector Machine and Random Forest classifiers along with Extreme Gradient Descent, Logistic Regression and K-Nearest Neighbors to try and correctly predict the absenteeism category of a given individual. 
