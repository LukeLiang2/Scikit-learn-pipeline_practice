# Scikit-learn_pipeline_practice

Here, I'll use Scikit learn pipeline for machine learning models building. I like to use nested cross validation which combines pipeline and grid search together. I think there are good reasons that using a pipeline is beneficial. 

1. **Avoid data leakage (most important)**

Many people do the preprocessing before the modeling, such as standardization or normalization. This is incorrect because using the entire dataset means data leakage. The pipeline can prevent this issue by applying preprocessing to the training set only.

2. **Grid search preprocessing and model**

Grid search is not only for the model's hyperparameter but also for preprocessing steps. For example, RobustScaler. What if I don't know the best quantile range? I can choose multiple of them like (25,75) and (10,90). This is highly convenient. 

3. **Less code compared to traditional fit&transform**

Traditional fit and transform will require lots of code because the "fit" can only be applied to the training set. Otherwise, data leakage issue appears. What if there are multiple preprocessing steps? It will require more code by keep doing fit (training set) and then transform (training and test set).
