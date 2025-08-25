#Task 1 : Perform EDA and Preprocessing 
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.metrics import mean_squared_error , confusion_matrix , classification_report
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.neighbors import KNeighborsClassifier



# Load dataset 

Data = fetch_california_housing(as_frame=True)
df = Data.frame

# define features and target 

X = df[["MedInc", "HouseAge", "AveRooms"]]
y = df["MedHouseVal"]

# Inspect data 

print(df.info())
print(df.describe())

# check missing values
print("Missing values: ", df.isnull().sum())

# Visualize relationships 
sns.pairplot(df , vars=["MedInc", "HouseAge", "AveRooms", "MedHouseVal"])
plt.show()

#split dataset
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2,random_state=42)

#Train linear reg
model = LinearRegression()
model.fit(X_train,y_train)

#make pred
y_pred = model.predict(X_test)

#evalute performance 
mse = mean_squared_error(y_test, y_pred)
print("Linear Regression MSE: ", mse)

#Task 2 : Train And evaluate multiple models 

#Load telco customer churn dataset 

df_telco = pd.read_csv("Telco-Customer-Churn.csv")

# Inspect data
print(df_telco.info())
print(df_telco.describe())

# check missing values
print("Missing values: ", df_telco.isnull().sum())

#handle missing values
df_telco.fillna(df_telco.mean(), inplace=True)

#Visualize Chuuurn  distribution 
sns.countplot(x="churn", data=df_telco)
plt.title("Churn Dis")
plt.show()

#Encode categorical variables 

le = LabelEncoder() 
df_telco['churn'] = le.fit_transform(df_telco['churn'])
df_telco['gender'] = le.fit_transform(df_telco['gender'])
df_telco['contact_type'] = le.fit_transform(df_telco['contact_type'])
df_telco['payment_method'] = le.fit_transform(df_telco['payment_method'])

#define features and target 
X = df_telco.drop(['churn'])
y = df_telco['churn'] 

# scale features 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split dataset 
X_train, X_test, y_train , y_test = train_test_split(X_scaled, y, test_size= 0.2 , random_state = 42)

# Train Logistic regression 
Log_model = LogisticRegression(max_iter=200)
Log_model.fit(X_train, y_train)

#train knn Model 
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

#evalute models 

log_pred = Log_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

print("\n Logistic Regression Clasification report:")
print(classification_report(y_test, log_pred))

print("\n k-NN  Clasification report:")
print(classification_report(y_test, knn_pred))

# confusion matrix for logistic regression 
print("confusion matrix for logistic regression", confusion_matrix(y_test, log_pred))
