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

We implemented the basic [logistic regression](https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/Code/DSML_Assignment_LogReg.py) algorithm without any additional specifications and got an accuracy of almost 30%. We then used Bayesian optimisation for hyperparameter tuning which increased accuracy to 39,78%. This is the highest value we obtained with the standard logistic regression. As can be seen below, we specified possible parameters for the regularisation strength *C*, the type, i.e. *penalty*, and the *solver*. Since the *lbfgs* solver does not support lasso regression, we tried two different parameter sets. We once excluded *L1* and used both solvers and once excluded *lbfgs* and used both, ridge and lasso regression. The resulting accuracies were almost identical with 39,8% (excluding *L1*) and 39,9% (excluding *lbfgs*). Since the latter is higher, we reported the final value of **39,9%** accuracy of a logistic regression model using regularisation strength *C* of 0.441, *L2* regularisation (Ridge Regression), and *liblinear* solver. 
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
*max_depth* is the maximum allowed depth of the tree. We specified it to be at 50 levels. While deeper trees can model more complex relationships in the data, they may also lead to overfitting if they are too deep relative to the complexity of the dataset. The same argumentation is valid for *min_samples_split* which indicates how many samples must accumulate at one node in order for it to split. Early splitting can namely lead to overfitted models. Similarly, *min_samples_leaf* specifies how many samples a leaf must have at least. This ensures the model does not learn overly specific patterns at the loss of generalisation. The different measures of quality of a split are given in criterion, where Gini Impurity and Entropy are the most common. In our optimal decision tree model, we use Gini Impurity, a maximum depth of 50 levels, a minimum amount of samples per leaf of 15, and a minimum amount of samples for a split of 20. This leaves us with an accuracy of **28.33%**. The remaining key evaluation metrics are given in the Table 1 above. 

The decision tree model performs substantially worse than logistic regression and KNN. This can be due to several reasons. Most likely, it is because in text analysis we employ TF-IDF vectorizers. These transform the text into TF-IDF features that are high-dimensional and sparse. Decision trees may not handle this kind of data well, as they make splits based on individual features, and many features in our case may have zero values.

#### Random Forest
<img width="989" alt="image" src="https://github.com/GiammarcoBozzelli/DSML/assets/55870958/1e42321e-b5d0-47da-bb7e-e677e2e93ca9">

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
The only parameter differing from the decision tree implementation is the *n_estimators*. Since we are building not a single but several trees in this approach, we need to specify how many we want to allow. Accordingly, n_estimators determines the number of individual decision trees that will be built and combined to form the random forest model. Allowing for several trees that then vote on an outcome seems to pay off. The accuracy of our optimised random forest classifier lies with **38,8%** substantially above the decision tree employed above. The optimal parameter values are Gini impurity, 200 estimators, a maximum depth of 50 levels, at least 1 sample per leaf, and at least 19 samples to allow a split.  

#### Support Vector Machine
<img width="733" alt="image" src="https://github.com/GiammarcoBozzelli/DSML/assets/55870958/9b62bfb5-f489-4d1d-a143-20eed3fb17f6">

Our first non-basic model is an [SVM model](https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/Code/DSML_Assignment_SVM.py) which are often used for text classification. It is suitable because of its effectiveness in high-dimensional spaces, as is the case with TF-IDF features, and its ability to find a hyperplane that best separates different classes. It essentially takes a set of labeled training data and tries to find the optimal hyperplane that separates the classes with the maximum margin (distance between the hyperplane and the nearest support vector from either class). Support vectors are the data points that are closest to the hyperplane. The hyperplane is the decision boundary that separates different classes in the feature space.
```
param_dist = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'kernel': Categorical(['linear', 'rbf', 'poly']),
    'degree': Integer(2, 5),   # Only relevant for 'poly' kernel
    'gamma': Categorical(['scale', 'auto'])  # Only relevant for 'rbf' and 'poly' kernels
}
```
As for the models above, we used Bayesian optimisation for hyperparameter tuning. The main parameters are the regularisation strength *C*, and the *kernel* type. Degree and gamma are further specifications for the kernel function which we will not explore further. *C* controls the trade-off between maximizing the margin and minimizing the classification error. A smaller value allows for a larger margin at the cost of more classification errors, leading to a softer margin. A larger value aims to classify all training examples correctly but may result in a smaller margin, leading to a harder margin. The *kernel* parameter specifies which function to use. That is a linear, radial basis (Gaussian), or polynomial kernel function. In our optimised model specification, we use a C value of 928.74 and a Gaussian kernel. This yields us a final accuracy of **43,65%** which is so far the highest attained accuracy for our language difficulty predictor.

#### Neural Network
![image](https://github.com/GiammarcoBozzelli/DSML/assets/55870958/7f7b3ee3-0e24-4a7a-a327-cc70a6788a64)

We then tried a [neural network](https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/Code/DSML_Assignment_NeuralNet.py) approach. It is built in four layers. The first is the embedding layer that converts the input sequences of word indices into vectors of fixed length. The second is the only hidden layer that processes the data and captures dependencies. The third layer is the dropout layer that prevents overfitting by randomly setting a fraction of input units to 0 (during training). The fourth is the output layer mapping the hidden layer's outputs to the available number of classes. We use Bayesian optimisation for hyperparameter tuning. The main goal is to define the dimension of the embedding (*embedding_dim*), the amount of neurons employed (*lstm_units*) in the hidden layer, and the dropout rate (*dropout_rate*). After implementing the tuner, we are left with an optimal model. It has an embedding dimension of 128, 64 neurons in the hidden layer, a dropout rate of 0.3, and a learning rate of 0.00092. Interestingly, the neural network approach only yields an accuracy of **45,1%**. This is only slightly better than the SVM approach above. 

dsf

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
