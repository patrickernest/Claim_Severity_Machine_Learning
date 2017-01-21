# Kaggle Project on Claim Severity

### DESCRIPTION

This project is a Kaggle project that we chose to do for our 5^th^ Machine Learning Project(CS 514 : Applied Artificial Intelligence at UIC). As you can see, the heading suggests that we need to predict Claim Severity. The given dataset has categories and continuous columns and a loss column. The loss is to be predicted. The categories and continuous
column are supposed to be used for predicting the loss.

Let us discuss a little on the purpose and scope of the project. Basically claim severity is the loss that is incurred from an insurance claim. Finance companies generally use traditional methods to predict loss. But after the recession in 2008, there is steep increase in severity for claims. As you can see in the graph below, the claim severity is increasing steeply between the years 2011-2015.

One solution for finance companies would be to design predictive models to accurately predict losses so that appropriate premiums can be determined for customers and there is no loss.

Before we get into have we can design a predictive model, we need to first install few packages first.

### REQUIRED INSTALLATION

* Python 3.5
* pandas
* numpy
* scikit-learn

(Please install the above packages before running the code. If you are using a Windows OS, use Anaconda to install. If you are using a Linux OS, use pip or easy\_install to install)

### PROCEDURE

#### INTRODUCTION TO THE GIVEN DATA AND DATA TRANSFORMATION

##### Data

The given data has totally 132 columns and has 188318 rows. Thisconsists of totally 116 categorical data and 14 continuous data columns, a loss column which has to be predicted and an id column which can be   disregarded.

##### Transformation

It is very important to convert the given categorical data into continuous data. This is important because ML models perform better in the Euclidean space. The given data is in the form of strings, and we are going to convert this into continuous data. There are many ways to transform this data into continuous values. I have tried two methods to
transform data, they are:-

1\) Assigned unique numbers to every category that appears in the dataset

2\) Used a scikit-learn library called One Hot Encoding which transforms the data into 0’s and 1’s

##### Scaling

I have also used a preprocessing library from scikit-learn, preprocessing.scale, where the variables are scaled to another dimension for better performance on ML models.

#### FEATURE SELECTION AND LEARNING

##### Feature Selection

Feature selection is an essential step in some scenarios because it helps for the model to predict more accurately considering only the needed features for predictions. I have also tried two methods for feature selection, they are:-

**SelectKBest** :- This is a library from scikit-learn which considers
only K number of feature based on the p-values calculated. This step
takes place before training the model.

**RFE**: - Recursive Feature Elimination is basically a method which is
used while the model is being trained. The idea is to repeatedly
construct a model and choose both the best performing feature (based on
coefficients) and eliminating the rest of the features.

##### Learning

Let us get into training a machine learning model to make predictions. I have tried several ML models for prediction, they are:-

* Linear Regression
* Lasso Regression
* Ridge Regression
* ElasticNet Regression
* K - Neighbors Regression
* Bayesian Ridge Regression
* Bagging Regression
* Extra Tree Regression
* Adaboost Regression
* Gradient Boosting Regression
* Random Forest Regression
* XGBoost Regression

####3 Evaluation Metrics

The evaluation metrics that I have used are:-

* 10-fold cross validation
* 80% training and 20% testing data

##### Running different iterations

I have run several iterations of the code where I have tried both Regular and One Hot Encoding Transformations with and without feature selection. The iterations are:-

**Results for best iteration**
One Hot Encoding with RFE and scaling gave me the best results which are:-
  ----------------------- ---------
  **ML Techniques**       **MAE** <br />
  Lasso Regression        1300.36 <br />
  Ridge Regression        1302.33 <br />
  Linear Regression       1303.44 <br />
  ElasticNet Regression   1293.66 <br />
  ExtraTree Regression    1331.3 <br />
  AdaBoost Regression     1893.08 <br />
  GradientBoost           1240.56 <br />
  XGBoost Regression      1297.54 <br />
  ----------------------- ---------

##### Comparison
Below is a comparison of all the results that I got from all the iterations:-



#### RUNNING CODE

Besides all the exploration with data transformation techniques and ML models, I have considered both Regular and One Hot Encoding Transformation. The One Hot Encoding takes longer to run, so Regular Transformation is recommended. Also, Gradient Boosting Regression is the best ML model among other models. If you could take a look at the graph above, the MAE for Gradient Boosting is the lowest of all the result, which means that it is the best. Since this model is the best I have considered to only running Gradient Boosting for the final code. The other code in the file have been commented out, so if you would like to explore other techniques please feel free to output results for other models too.

##### Operating System and Hardware on which the code was initially run on
* Ubuntu 16.04(Linux Distro)
* Intel i7 Processor
* 8GB RAM

##### Things to keep in mind before running code
* Make sure you have install all the required packages for the code to run
* Make sure you have the train.csv file given in the Kaggle competition page available in the directory where the code is running

##### Running code (Use command)

```python3 claim_severity_pat.py```

##### Problems that maybe encountered during running

The runtime of the code may differ from computer to computer. The regular transformation part of the code is running in a 1min 34sec (on my system). And for the One Hot Encoding transformation of the code was running in 6mins 10sec (one my system). This can vary, in some cases it can take longer. Also, the when the data is split into 80% and 20% training and testing, the data is shuffled and the MAE and output can vary every iteration. Therefore, I have created an extra file so the actual values and predicted values can be compared.

##### Expected MAE

* Regular Transformation – MAE :- 1240 (+- 10)

* One Hot Encoding Transformation – MAE :- 1250 (+- 10)

### ABOUT PROJECT

This Project was done as a part of CS:514 Applied Artificial Intelligence at UIC.
