# Data Science and Machine Learning
## Final Report: Team Basel

### Intro
Bla bla 

### Models & Results
!!!INSERT TABLE 1 HERE!!!
The task was to... 
We used... 
This gave us... 
The best results yields ..., because of .... We got this by doing ...

#### Logistic Regression
We implemented the basic logistic regression algorithm without any additional specifications and got an accuracy of almost 30%. We then used Bayesian optimisation for hyperparameter tuning which increased accuracy to 39,78%. This is the highest value we obtained with the standard logistic regression. As can be seen below, we specified possible parameters for the regularisation strength **C**, the type, i.e. **penalty**, and the **solver**.  For the penalty, we chose to stick with ridge regression (l2), since the lbfgs solver does not support lasso regression. 
```
param_dist = {
    'C': (1e-6, 1e+6, 'log-uniform'),
    'penalty': ['l2'],           
    'solver': ['lbfgs', 'liblinear'],
}
```


#### k-Nearest Neighbours
We implemented the basic KNN model and played around with the parameters to get a feeling of how they behave. Generally, more neighbours do not necessarily increase accuracy. It depends on the weighting and distance metric employed. In our case, using cosine similarity as distance metric gave the highest accuracy. Using Bayesian optimisation, we found that the KNN model with 21 neighbours, cosine similarity, and distance-dependent weights gave the highest accuracy of 32,92%. The remaining common evaluation metrics are given in _Table 1_ above. The optimal values for our parameters were found using Bayesian optimisation for hyperparameter tuning. The parameters for which we wanted to find optimal values are given in the code snippet below. It includes the amount of neighbours, whether we assign uniform weights or dependent on distance, and the three most common metrics for distance measuring.
```
param_dist = {
    'n_neighbors': (1, 30),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'cosine']
}
```

### Application 
Link to the Webapp 
Explanation of the Webapp 
Limitations of the Webapp 
Show the use of the predictor!!!

### Video
Tutorial Link

### Work Partition 
Gimmy did
Tim did

### Disclaimer on GPT
We used chat only to help us code bla bla
