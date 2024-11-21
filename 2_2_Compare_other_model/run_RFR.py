import pandas as pd
data = pd.read_csv("PaDELPy_Data.csv")
data=data.iloc[:, 1:]

from sklearn.feature_selection import VarianceThreshold
import numpy as np

selector = VarianceThreshold(threshold=0.15)
df_reduced = selector.fit_transform(data)
columns_retained = data.columns[selector.get_support()]
data_reduced = pd.DataFrame(df_reduced, columns=columns_retained, index=data.index)
print(data_reduced.shape)

corr_matrix = data_reduced.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
data_reduced = data_reduced.drop(columns=to_drop)
data_reduced = data_reduced.dropna(axis=1)

print(data_reduced.shape)


##

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = data_reduced.iloc[:, :-1]
y = data_reduced.iloc[:, -1]

scaler = StandardScaler()
X_ = scaler.fit_transform(X)


# oversample = SMOTE()
# X_, y = oversample.fit_resample(X_, y)


X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2, random_state=2024)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500], 
    'max_depth': [10, 15, 20, 25, 30, None],  
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],  
    'max_features': ['sqrt', 'log2', 0.2, 0.5],  
    'bootstrap': [True],  
    'max_samples': [None, 0.5, 0.75, 1.0]  
}

rf = RandomForestRegressor()

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

import joblib

# 保存模型到文件
joblib.dump(best_model, 'best_random_forest_model.pkl')


# 使用最佳参数训练模型
# best_model = grid_search.best_estimator_

# rf = RandomForestRegressor(
#     n_estimators=200,
#     max_depth=20,
#     min_samples_split=10,
#     min_samples_leaf=5,
#     max_features='sqrt',
#     bootstrap=True,
#     random_state=2024
# )
# rf.fit(X_train, y_train)
y_pred_rf = best_model.predict(X_test)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

print(round(rf_mse,3),round(rf_mae,3),round(rf_r2,3))