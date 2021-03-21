# Machine Learning: Exoplanet Exploration

Data and instructions provided by UC Berkeley Extension Data Analytics Bootcamp.

# Introduction 

The goal of this assignment is to use my newfound knowledge and skills on machine learning to create models capable of classifying candidate exoplanets from the raw dataset.

# Reporting

#### Comparison between the KNN and SVM models:

Both models yielded the same prediction accuracy rate, which is 87%. Nevertheless, the processes of feature selections and fine tuning are different for each model. To elaborate, for the KNN model, it yielded a higher accuracy rate if I select features with higher rating (> 0.08) after using the Random Forest Classifier for examination. On the other hand, for the SVM, it did not need the Random Forest Classifier to identify the more relevant features. It yielded a higher accuracy when I used all features.

The hyperparameter tuning process was also different for both models. For the KNN model, the accuracy improved when I increased the knn range from 20 to 50. For the SVM mode, I used 'C':[1, 5, 10], 'gamma': [0.0001, 0.001, 0.01] to get the current accuracy.

#### Summary

I think my model would be good enough to predict new exoplanets with prediction accuracy of 87%. The hyperparameter tuning process did improve both models' accuracy. I think there are still many things I could do and try to improve my model accuracy, such as spending more time to address the SVM model's parameter grid for hypyerparameter tuning, and trying to add more parameters. For the KNN model, I could also try adjusting the hyperparameter tuning to improve the accuracy. Last but not least, it is likely that these two models that I created weren't the most suitable model to give us an amazing prediction accuracy. 

# Technologies/Libraries

- python

- sklearn

- pandas

- matplotlib

- numpy

- joblib

- conda environment: PythonAdv

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
    
    - https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    - https://www.ritchieng.com/machine-learning-efficiently-search-tuning-param/
    - https://data-flair.training/blogs/svm-kernel-functions/
    - https://www.kdnuggets.com/2016/06/select-support-vector-machine-kernels.html
    - https://www.baeldung.com/cs/svm-multiclass-classification
    - https://stackoverflow.com/questions/55314345/how-to-find-feature-importance-or-variable-importance-graph-for-knnclassifier
    - https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
