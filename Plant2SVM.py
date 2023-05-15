import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


#load data
P2 = pd.read_csv('Plant2_summed_dataset.csv', na_values= '?')

P2 = P2[['DATE_TIME', 'AC_POWER','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION']]

P2_det=P2.copy()
anomaly_inputs=['AC_POWER','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION']
#IN IsolationForest, adjust contamination [0-0,5], indicates the percentage of expected outliers.
#Use plot at the end of this code block to check if the outliers are actually identified
model_IF = IsolationForest(contamination=0.03, random_state=42)
model_IF.fit(P2_det[anomaly_inputs])
P2_det['anomaly_scores'] = model_IF.decision_function(P2_det[anomaly_inputs])
P2_det['anomaly'] = model_IF.predict(P2_det[anomaly_inputs])


#OUTLIER INDEXES
out_index=np.asarray(np.where(P2_det['anomaly']==-1)[0]).astype(int)
#INLIER INDEXES
in_index=np.asarray(np.where(P2_det['anomaly']==1)[0]).astype(int)

#MAKING COPY TO MAKE CHANGES TO
P2_imp=pd.DataFrame(P2.copy())

#REPLACING OUTLIERS WITH NaN
P2_imp.iloc[out_index,1]=np.nan
P2_imp['DATE_TIME']=pd.to_datetime(P2_imp['DATE_TIME'])



#KNN IMPUTATION
imputer = KNNImputer(n_neighbors=3)
P2_imp.iloc[:,1:]=imputer.fit_transform(P2_imp.iloc[:,1:])


#print('missing values:',P2_imp.isna().sum())


plt.figure(figsize=(14,5))
start, end= 0,(len(P2)-1)
plt.plot(pd.to_datetime(P2_det.iloc[:,0]),P2_det.iloc[:,1],color='green')#ORIGINAL DATA
plt.plot(pd.to_datetime(P2_det.iloc[in_index,0]),P2_det.iloc[in_index,1],color='red')#INLIERS
plt.plot(pd.to_datetime(P2_det.iloc[:,0]),P2_imp.iloc[:,1],color='purple')#IMPUTED DATA
plt.scatter(pd.to_datetime(P2_det.iloc[out_index,0]),P2_det.iloc[out_index,1],color='blue')#OUTLIERS
# plt.plot(pd.to_datetime(P2_det.iloc[:,0]),P2_det.iloc[:,1+3]*20000,color='orange')#For plotting Irradiation alongside
plt.xlim([pd.to_datetime(P2_det.iloc[start,0]),pd.to_datetime(P2_det.iloc[end,0])])
plt.legend(['Original data','Inliers','Imputed data','Outliers'])
# plt.legend(['Original data','Inliers','Imputed data','Outliers','Irradiation'])##For plotting Irradiation alongside
plt.show()

#USE THIS DATA INSTEAD IN THE REST OF THE CODE
feature_names=['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
X_imp = P2_imp[feature_names]
y_imp = P2_imp['AC_POWER']
# scaler1=MinMaxScaler()
scaler1=StandardScaler()
scaler1.fit(X_imp)
X_imp=pd.DataFrame(scaler1.transform(X_imp))
# X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X_imp, y_imp, test_size=6/34, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X_imp, y_imp, test_size=6/34, shuffle=False)
# display(P2_imp)





#The svm regressor
svm = SVR()

#Train the model 
svm.fit(X_train, y_train)

#Make predictions on the test data
y_pred = svm.predict(X_test)

# Gridsearch and  Fit the GridSearchCV object to the training data
param_grid = {'C': [50 ,100, 150, 170, 180, 190, 200],
              'gamma': [1.5, 1, 0.7, 0.6, 0.5, 0.1, 0.01],
              'kernel': ['rbf']}

# TimesSeriesSplit
tscv = TimeSeriesSplit(n_splits=7)
grid = GridSearchCV(svm, param_grid, cv=tscv, return_train_score=True)
grid.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best hyperparameters: ", grid.best_params_)
print(grid.score(X_test, y_test))


# calculate feature importances
results = permutation_importance(grid.best_estimator_, X_test, y_test, n_repeats=10, random_state=0)
importances = results.importances_mean

# create a bar plot of feature importances
plt.bar(feature_names, importances)
plt.xlabel('Feature')
plt.ylabel('Relative importance')
plt.show()


# Predict on test data
y_pred = grid.best_estimator_.predict(X_test)

y_test_arr = y_test.to_numpy()

NN = y_test_arr.size
# print(NN)
time_step = 0.25 #  in hours
new_time_axis = np.linspace(0, NN*time_step-1*time_step, NN)
print(new_time_axis.size)

plt.figure(figsize=(14,5))
plt.plot(new_time_axis, y_test_arr, label="y_test")
plt.plot(new_time_axis, y_pred, label="y_predict")
plt.legend()
plt.title("AC Power Prediction with Support Vector Regressor")
plt.ylabel("AC Power $[kW]$")
plt.xlabel("Time [h]")
plt.grid()
days = 6
plt.xticks(np.arange(0, 24*(days+1), step=24))  # Set label locations.
plt.xlim(24*(0), 24*(days-0))    
plt.ylim(0,max(y_test_arr)*1.05)
plt.show()

SVR_score_train = grid.best_estimator_.score(X_train, y_train)
print(f"Train score is {SVR_score_train*100:.4f} %")

SVR_score_test = grid.best_estimator_.score(X_test, y_test)
print(f"Test score is {SVR_score_test*100:.4f} %")



results = pd.DataFrame(grid.cv_results_)
pivot_table = pd.pivot_table(results, values='mean_test_score', index='param_gamma', columns='param_C')

# Create the heatmap
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt=".4f")
plt.xlabel('C')
plt.ylabel('Gamma')
plt.title('SVR Grid Search Accuracies')
plt.show()
