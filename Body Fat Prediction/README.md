# Body Fat Prediction (Regression)

https://www.kaggle.com/fedesoriano/body-fat-prediction-dataset


**Context**

Lists estimates of the percentage of body fat determined by underwater
weighing and various body circumference measurements for 252 men.

**Educational use of the dataset**

This data set can be used to illustrate multiple regression techniques. Accurate measurement of body fat is inconvenient/costly and it is desirable to have easy methods of estimating body fat that are not inconvenient/costly.

**Pipeline + Nested cross-validation with hyperparameter tuning

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler,PowerTransformer
from sklearn.pipeline import make_pipeline; from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsRegressor

numcol = x_train.select_dtypes(include=['int64', 'float64']).columns
power = PowerTransformer(method='yeo-johnson')
rob = RobustScaler(quantile_range=(25.0, 75.0))
ct = make_column_transformer((make_pipeline(power,rob),numcol),remainder='passthrough')
knn = KNeighborsRegressor(n_jobs=-1)

pipeline = make_pipeline(ct,knn,verbose=False)
