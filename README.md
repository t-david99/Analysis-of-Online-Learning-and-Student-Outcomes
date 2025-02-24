# Analysis of Online Learning and Student Outcomes

## Overview 

In comparison to the traditional brick-and-mortar educational environment, the online learning environment has been on the rise, especially after the COVID-19 pandemic struck the world. Additionally, the online environment has recently seen a significant rise in dropout rates (He et al, 2020). With an improvement in online learning, people all over the world can have access to a high-quality education. A high-quality education opens opportunities for people from anywhere, since it can be difficult to find a good education in some places.

I aim to accurately make predictions on whether a student will dropout based on the given data, and all factors impacting this prediction will be explored. The dependent variable is categorical with three possible classes: dropout, enrolled, and graduate. A challenge lies in the fact that the distribution of the response variable is quite imbalanced. This can easily result in a biased model towards the most common class among the responses. Alleviation of this issue will involve introducing class weights to each of the classification models, leading to better predictions. Furthermore, there is a large number of features (especially once dummy variables are introduced). Multiple different dimension-reducing methods will be used including t-stochastic neighbor embedding (t-SNE) for data visualization. Singular value decomposition (SVD) and LASSO regression will be used for dimensionality reduction for the use of the classification models (random forest, AdaBoost, support vector machine, k-nearest neighbors, and logistic regression).

## Project Files

### 1. Data Set

I obtained the data set (data.csv in the project folder) that is used throughout this project from the UCI Machine Learning Repository (Link to the data set: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success). Each data point is an anonymous student who is enrolled in one out of multiple different undergraduate degrees. 4424 students are included in this data set in total. Furthermore, this data set includes a total of 36 predictors. 17 of which are categorical variables named "Mother's qualification", "Father's qualification","Mother's occupation", "Father's occupation", "Marital status", "Nacionality", "Application mode", "Course", "Gender", "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date", "Scholarship holder", "International", and "Daytime/evening attendance". All were converted into dummy variables using the Pandas library in Python. The response variable was also categorical, indicating either dropout, enrolled, or graduated. The remaining 19 continuous variables in this data set include "Admission grade", "Previous qualification (grade)", "Unemployment rate", "Inflation rate", "GDP", number of credits taken for each semester, and "Age at enrollment".

### 2. Code

- Dropout_Project_EDA_Code.ipynb contains an exploratory data analysis performed on the data set. EDA was performed to detect multicollinearity, and to look at the data's structure, summary statistics, and feature distributions.

- Dropout_Project_All_Predictors_Code.ipynb contains the fitting and testing of multiple classification models, using all predictors. It also includes some analysis of the results.

- Dropout_Project_LASSO_Code.ipynb contains the fitting and testing of multiple classification models, using predictors chosen by the LASSO regression model, which was also evaluated. It also includes some analysis of the results.

- Dropout_Project_SVD_Pred_Code.ipynb contains the fitting and testing of multiple classification models, using the predictors created through singular value decomposition of the data that explain 95% of the variance of the original data. It also includes some analysis of the results.

### 3. Results/Conclusions

- Dropout_Project_Conclusions.txt is a text file that summarizes the project's findings by comparing different models based on various metrics and highlighting key factors influencing student dropout rates.

## How to Run the Project

### 1. Dependencies

Using the following code, install the following Python libraries (if you haven't already): pip install pandas matplotlib seaborn scikit-learn

### 2. Notebook Execution

Using your favorite IDE, run the Python notebooks in the following order:

1. Dropout_Project_EDA_Code.ipynb

2. Dropout_Project_All_Predictors_Code.ipynb

3. Dropout_Project_LASSO_Code.ipynb

4. Dropout_Project_SVD_Pred_Code.ipynb

### 3. Analysis of the Results

A thorough analysis of the results and how they impact the problem at hand can be found in the Dropout_Project_Conclusions.txt file.

## References

He, Y., Chen, R., Li, X., Hao, C., Liu, S., Zhang, G., and Jiang, B. (2020). Online at-risk student identification using 
rnn-gru joint neural networks. Information, 11(10)
