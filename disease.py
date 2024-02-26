import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns




data = pd.read_csv("heart_dataset.csv")

data.columns

data.head()

data.tail()
data.info()


# Checking for Null Values (0 indicates there is no null value)

data.isnull().sum()


data.shape

data.describe()


# Transposed Statistical Description

data.describe().T


# How many CAD and non-CAD Patients are there ?



data['target'].value_counts()


# Visualizing the output 



data['target'].value_counts().plot(kind='bar', figsize=(9,9), color= ['red','blue'])
plt.minorticks_on()
plt.grid(which='major',color='coral',linestyle=':')
plt.xlabel('CAD and Non-CAD states')
plt.ylabel('Patient Count')
plt.title('CAD and Non-CAD patient count')



# The correlation between variables or attributes which causes CAD The most

# cholesterol is the leading cause of CAD

data.corr()['cholesterol'].sort_values().plot(kind='bar')


# The numbers of outliers in each attribute


Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)

IQR = Q3 - Q1

((data < (Q1 - 1.5 * IQR)) | (data < (Q3 - 1.5 * IQR))).sum()


# **The Distribution of Continuous values**




for feature in data:
  dataset = data.copy()
  dataset[feature].hist(bins=25)
  plt.xlabel(feature)
  plt.ylabel("Count")
  plt.title(feature)
  plt.show()


# Detecting Outliers using z_score 



outliers = []
def detect_outliers(values):
  Threshold = 3
  mean_val = np.mean(values)
  std_val = np.std(values)

  for i in values:
    z_score = (i-mean_val)/std_val
    if np.abs(z_score) > Threshold:
      outliers.append(i)
  return outliers

out = detect_outliers(data['age'])
out



outliers = []
def detect_outliers(values):
  Threshold = 3
  mean_val = np.mean(values)
  std_val = np.std(values)

  for i in values:
    z_score = (i-mean_val)/std_val
    if np.abs(z_score) > Threshold:
      outliers.append(i)
  return outliers

out = detect_outliers(data['resting bp s'])
out


outliers = []
def detect_outliers(values):
  Threshold = 3
  mean_val = np.mean(values)
  std_val = np.std(values)

  for i in values:
    z_score = (i-mean_val)/std_val
    if np.abs(z_score) > Threshold:
      outliers.append(i)
  return outliers

out = detect_outliers(data['cholesterol'])
out


outliers = []
def detect_outliers(values):
  Threshold = 3
  mean_val = np.mean(values)
  std_val = np.std(values)

  for i in values:
    z_score = (i-mean_val)/std_val
    if np.abs(z_score) > Threshold:
      outliers.append(i)
  return outliers

out = detect_outliers(data['max heart rate'])
out


# The correlation between features (Heat Chart)



plt.figure(figsize=(14,10)) 

sns.heatmap(data.corr(),annot=True,cmap='hsv',fmt='.3f',linewidths=2)

plt.ylim(12,0) 

plt.show()


# Splitting Dataset into Train and Test Parts

from sklearn.model_selection import train_test_split

X = data.drop('target', axis=1).values
y = data['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# Scaling Data


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ML MODELS

# 1- Logistic Regression Algorithm (LR)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg = LogisticRegression(max_iter=1200)
logreg.fit(X_train, y_train)
lr_predictions = logreg.predict(X_test)

accuracy = accuracy_score(y_test, lr_predictions)
print(f"Logistic Regression Accuracy: {accuracy}")

print(confusion_matrix(y_test,lr_predictions))


# 2- Support Vector Machines Algorithm (SVM)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)

accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {accuracy}")


print(confusion_matrix(y_test,svm_predictions))


# 3- K-Nearest Neighbors Algorithm (KNN)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

accuracy = accuracy_score(y_test, knn_predictions)
print(f"KNN Accuracy: {accuracy}")
print(confusion_matrix(y_test,knn_predictions))


# 4- Decision Tree Algorithm (DT)


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_predictions = dt.predict(X_test)

accuracy = accuracy_score(y_test, dt_predictions)
print(f"Decision Tree Accuracy: {accuracy}")

print(confusion_matrix(y_test,dt_predictions))


# 5- Random Forest Algorithm (RF)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)

accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {accuracy}")


print(confusion_matrix(y_test,rf_predictions))


# 6- Gradient Boosting Classifier Algorithm (GBC)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

gbc_predictions = gbc.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, gbc_predictions))


print(confusion_matrix(y_test,gbc_predictions))


# choosing Random forest
import joblib
joblib.dump(rf,"cad_model.pkl")


# **Model Testing**

# first test Pateint with No CAD


Model = joblib.load('cad_model.pkl')
Model.predict([[55,0,2,160,220,0,1,140,1,1.0,2]])


# second test Pateint with CAD

Model = joblib.load('cad_model.pkl')
Model.predict([[45,1,1,110,264,0,0,132,0,1.2,2]])


# 3rd test Pateint with No CAD

Model = joblib.load('cad_model.pkl')
Model.predict([[63,1,1,145,233,1,2,150,0,2.3,2]])


# 4th Pateint with CAD

Model = joblib.load('cad_model.pkl')
Model.predict([[67,1,4,160,286,0,2,108,1,1.5,2]])

# 5th test patient with no CAD
model= joblib.load('cad_model.pkl')
model.predict([[48,0,2,120,284,0,0,120,0,0,1]])