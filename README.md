# Exoplanet Exploration

Data and instructions provided by UC Berkeley Extension Data Analytics Bootcamp.

# Introduction 

The goal of this assignment is to use my newfound knowledge and skills on machine learning to create models capable of classifying candidate exoplanets from the raw dataset.

# Technologies/Libraries

- python

- sklearn

- pandas

- matplotlib

- numpy

- joblib

# Detailed Instructions/Assignment Background

### Background

    Over a period of nine years in deep space, the NASA Kepler space telescope has been out on a planet-hunting mission to discover hidden planets outside of our solar system.

    To help process this data, you will create machine learning models capable of classifying candidate exoplanets from the raw dataset.

### Instructions

##### Preprocess the Data
    - Preprocess the dataset prior to fitting the model.
    - Perform feature selection and remove unnecessary features.
    - Use MinMaxScaler to scale the numerical data.
    - Separate the data into training and testing data.

##### Tune Model Parameters

    - Use GridSearch to tune model parameters.
    - Tune and compare at least two different classifiers.

##### Reporting

    Create a README that reports a comparison of each model's performance as well as a summary about your findings and any assumptions you can make based on your model (is your model good enough to predict new exoplanets? Why or why not? What would make your model be better at predicting new exoplanets?).

# Files

    - README.md contains my reporting in addition to assignment details
    - exoplanet_data.csv is the data I used
    - model_1_KNN.ipynb is the jupyter notebook I used to create the KNN model
    - model_2_SVM.ipynb is the jupyter notebook I used to create the SVM model
    - yuying.sav is the file that contain the file of my best model (both models yield the same accuracy, I randomly picked model 2)
    
# Process and Credits

My first assignment working with sklearn to create machine learning model. I used class materials and outside resources for reference. 

Here are the outside resources that I used for this assignment (as well as attempts):

    - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    - https://www.ritchieng.com/machine-learning-efficiently-search-tuning-param/
    - https://data-flair.training/blogs/svm-kernel-functions/
    - https://www.kdnuggets.com/2016/06/select-support-vector-machine-kernels.html
    - https://www.baeldung.com/cs/svm-multiclass-classification
    - https://stackoverflow.com/questions/55314345/how-to-find-feature-importance-or-variable-importance-graph-for-knnclassifier
    - https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
