## Plan


##### 11-03-2018
* Naive Bayes performs well when we have multiple classes and working with text classification. Advantage of Naive Bayes algorithms are:
* It is simple and if the conditional independence assumption actually holds, a Naive Bayes classifier will converge quicker than discriminative models like logistic regression, so you need less training data. And even if the NB assumption doesn’t hold.
* It requires less model training time
* The main difference between Naive Bayes(NB) and Random Forest (RF) are their model size. Naive Bayes model size is low and quite constant with respect to the data. The NB models cannot represent complex behavior so it won’t get into over fitting. On the other hand, Random Forest model size is very large and if not carefully built, it results to over fitting.
* So, When your data is dynamic and keeps changing. NB can adapt quickly to the changes and new data while using a RF you would have to rebuild the forest every time something changes.


* The advantages of support vector machines are:

    1. Effective in high dimensional spaces.
    2. Still effective in cases where number of dimensions is greater than the number of samples.
       Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
    3. Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also              possible to specify custom kernels.

*The disadvantages of support vector machines include:

    1. If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and                  regularization term is crucial.
    2. SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores        and probabilities, below).
    
    
 #### SVC
 * The C parameter trades off misclassification of training examples against simplicity of the decision surface. A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly by giving the model freedom to select more samples as support vectors.
