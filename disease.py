import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter


data = pd.read_csv("heart_dataset.csv")


data.columns

data.head()

data.tail()
data.info()


data.isnull().sum()




data.shape




data.describe()




# Transposed Statistical Description
data.describe().T




data['target'].value_counts()




# Visualizing the output 
data['target'].value_counts().plot(kind='bar', figsize=(9,9), color= ['red','blue'])
plt.minorticks_on()
plt.grid(which='major',color='coral',linestyle=':')
plt.xlabel('CAD and Non-CAD states')
plt.ylabel('Patient Count')
plt.title('CAD and Non-CAD patient count')




# cholesterol is the leading cause of CAD
data.corr()['cholesterol'].sort_values().plot(kind='bar')



# The numbers of outliers in each attribute
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)



IQR = Q3 - Q1

((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()


# # The Distribution of Continuous values



for feature in data:
  dataset = data.copy()
  dataset[feature].hist(bins=25)
  plt.xlabel(feature)
  plt.ylabel("Count")
  plt.title(feature)
  plt.show()


# # Detecting Outliers using z_score 



import numpy as np

def detect_outliers(data, feature):
    values = data[feature]
    mean_val = np.mean(values)
    std_val = np.std(values)
    outliers = []

    Threshold = 3
    for i in values:
        z_score = (i - mean_val) / std_val
        if np.abs(z_score) > Threshold:
            outliers.append(i)
    return outliers



age_outliers = detect_outliers(data, 'age')
print(f"Age Outliers: {age_outliers}")




bp_outliers = detect_outliers(data, 'resting bp s')
print(f"Resting BP 's' Outliers: {bp_outliers}")


cholesterol_outliers = detect_outliers(data, 'cholesterol')
print(f"Cholesterol Outliers: {cholesterol_outliers}")




max_heart_rate_outliers = detect_outliers(data, 'max heart rate')
print(f"Max Heart Rate Outliers: {max_heart_rate_outliers}")


# # The correlation between features (Heat Chart)



import seaborn as sns




plt.figure(figsize=(14,10)) 

sns.heatmap(data.corr(),annot=True,cmap='hsv',fmt='.3f',linewidths=2)

plt.ylim(12,0)




plt.show()


# # SMOTE



X = data.drop('target', axis=1)
y = data['target']



# Summarize class distribution
print("Before SMOTE:", Counter(y))



oversample = SMOTE(random_state=42)
X_resampled, y_resampled = oversample.fit_resample(X, y)



# Summarize the new class distribution
print("After SMOTE:", Counter(y_resampled))



# Splitting Dataset into Train and Test Parts
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=2)


# # Scaling Data



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # ML MODELS

# 1- Logistic Regression Algorithm (LR)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix




X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=2)




logreg = LogisticRegression(max_iter=1200)
logreg.fit(X_train, y_train)
lr_predictions = logreg.predict(X_test)




accuracy = accuracy_score(y_test, lr_predictions)
print(f"Logistic Regression Accuracy: {accuracy}")



# 2- Support Vector Machines Algorithm (SVM)



from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=2)




svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)



accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {accuracy}")


# 3- K-Nearest Neighbors Algorithm (KNN)




from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score





X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=2)





knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)



accuracy = accuracy_score(y_test, knn_predictions)
print(f"KNN Accuracy: {accuracy}")



# 4- Decision Tree Algorithm (DT)



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score





X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=2)





dt = DecisionTreeClassifier(random_state=2)
dt.fit(X_train, y_train)
dt_predictions = dt.predict(X_test)





accuracy = accuracy_score(y_test, dt_predictions)
print(f"Decision Tree Accuracy: {accuracy}")


# 5- Random Forest Algorithm (RF)




from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score





X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=2)




rf = RandomForestClassifier(n_estimators=100, random_state=2)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)





accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {accuracy}")




# 6- Gradient Boosting Classifier Algorithm (GBC)




from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics





gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)





gbc_predictions = gbc.predict(X_test)





print("Accuracy:", metrics.accuracy_score(y_test, gbc_predictions))








# # Set up hyperparameter grid for tuning




from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier





param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}





grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)





# Fit the model
grid_search.fit(X_train, y_train)





# Best parameters
print(f"Best parameters: {grid_search.best_params_}")



# Evaluate the model with cross-validation
cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy Scores: {cv_scores}")




# Predict on the test set
y_pred = grid_search.predict(X_test)



# Model Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))


# # Error Analysis - Confusion Matrix


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


# # Feature Importance Analysis with Random Forest




importances = rf.feature_importances_


#Get the feature names
feature_names = data.drop('target', axis=1).columns




# Summarize feature importance
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)



print(feature_importances)


# Visualizing the feature importances
plt.figure(figsize=(10,8))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
