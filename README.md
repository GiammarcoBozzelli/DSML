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
<img width="358" alt="image" src="https://github.com/GiammarcoBozzelli/DSML/assets/55870958/3ea4dd8b-0fa8-48ee-8f18-e76a94df712d">

We implemented the basic [logistic regression](https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/Code/DSML_Assignment_LogReg.py) algorithm without any additional specifications and got an accuracy of almost 30%. We then used Bayesian optimisation for hyperparameter tuning which increased accuracy to 39,78%. This is the highest value we obtained with the standard logistic regression. As can be seen below, we specified possible parameters for the regularisation strength **C**, the type, i.e. **penalty**, and the **solver**. Since the lbfgs solver does not support lasso regression, we tried two different parameter sets. We once excluded L1 and used both solvers and once excluded lbfgs and used both, ridge and lasso regression. The resulting accuracies were almost identical with 39,8% (excluding L1) and 39,9% (excluding lbfgs). Since the latter is higher, we reported the final value of **39,9%** accuracy of a logistic regression model using regularisation strength C of 0.441, L2 regularisation (Ridge Regression), and liblinear solver. 
```
param_dist = {
    'C': (1e-6, 1e+6, 'log-uniform'),
    'penalty': ['l2', "l1"],           
    'solver': ["liblinear"],
}
```

#### k-Nearest Neighbours
We implemented the basic [KNN model](https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/Code/DSML_Assignment_KNN.py) and played around with the parameters to get a feeling of how they behave. Generally, more neighbours do not necessarily increase accuracy. It depends on the weighting and distance metric employed. In our case, using cosine similarity as distance metric gave the highest accuracy. Using Bayesian optimisation, we found that the KNN model with 21 neighbours, cosine similarity, and distance-dependent weights gave the highest accuracy of **32,92%**. The remaining common evaluation metrics are given in _Table 1_ above. The optimal values for our parameters were found using Bayesian optimisation for hyperparameter tuning. The parameters for which we wanted to find optimal values are given in the code snippet below. It includes the amount of neighbours, whether we assign uniform weights or dependent on distance, and the three most common metrics for distance measuring.
```
param_dist = {
    'n_neighbors': (1, 30),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'cosine']
}
```

#### Decision Tree
<img width="989" alt="image" src="https://github.com/GiammarcoBozzelli/DSML/assets/55870958/161fdc9d-dc28-49d6-bdd1-25e6d6864486">

As for logistic regression and KNN approaches above, we used bayesian optimisation for hyperparameter tuning to find the most promising values for the parameters of a standard [decision tree model](https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/Code/DSML_Assignment_DecisionTree.py). The specified optimisation is given below. 
```
param_dist = {
    'max_depth': Integer(1, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20),
    'criterion': Categorical(['gini', 'entropy']),
    'max_features': Categorical([None, 'sqrt', 'log2'])
}
```
max_depth is the maximum allowed depth of the tree. We specified it to be at 50 levels. While deeper trees can model more complex relationships in the data, they may also lead to overfitting if they are too deep relative to the complexity of the dataset. The same argumentation is valid for min_samples_split which indicates how many samples must accumulate at one node in order for it to split. Early splitting can namely lead to overfitted models. Similarly, min_samples_leaf specifies how many samples a leaf must have at least. This ensures the model does not learn overly specific patterns at the loss of generalisation. The different measures of quality of a split are given in criterion, where Gini Impurity and Entropy are the most common. In our optimal decision tree model, we use Gini Impurity, a maximum depth of 50 levels, a minimum amount of samples per leaf of 15, and a minimum amount of samples for a split of 20. This leaves us with an accuracy of **28.33%**. The remaining key evaluation metrics are given in the Table 1 above. 

The decision tree model performs substantially worse than logistic regression and KNN. This can be due to several reasons. Most likely, it is because in text analysis we employ TF-IDF vectorizers. These transform the text into TF-IDF features that are high-dimensional and sparse. Decision trees may not handle this kind of data well, as they make splits based on individual features, and many features in our case may have zero values.

#### Random Forest
As for the decision tree implementation above, we use hyperparameter tuning to find the most promising values for a prediction using a [random forest classifier](https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/Code/DSML_Assignment_RandomForest.py). 
```
param_dist = {
    'n_estimators': Integer(10, 200),
    'max_depth': Integer(1, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20),
    'max_features': Categorical([None, 'sqrt', 'log2']),
    'criterion': Categorical(['gini', 'entropy'])
}
```
The only parameter differing from the decision tree implementation is the n_estimators. Since we are building not a single but several trees in this approach, we need to specify how many we want to allow. Accordingly, n_estimators determines the number of individual decision trees that will be built and combined to form the random forest model. Allowing for several trees that then vote on an outcome seems to pay off. The accuracy of our optimised random forest classifier lies with **38,8%** substantially above the decision tree employed above. The optimal parameter values are Gini impurity, 200 estimators, a maximum depth of 50 levels, at least 1 sample per leaf, and at least 19 samples to allow a split.  

#### Support Vector Machine
Our first non-basic model is an [SVM model](https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/Code/DSML_Assignment_SVM.py) which are often used for text classification. It is suitable because of its effectiveness in high-dimensional spaces, as is the case with TF-IDF features, and its ability to find a hyperplane that best separates different classes. It essentially takes a set of labeled training data and tries to find the optimal hyperplane that separates the classes with the maximum margin (distance between the hyperplane and the nearest support vector from either class). Support vectors are the data points that are closest to the hyperplane. The hyperplane is the decision boundary that separates different classes in the feature space.

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
