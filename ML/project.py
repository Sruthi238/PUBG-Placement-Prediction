import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
col_names=['groupId','matchId','assists','boosts','headshotKills','heals']
dataset = pd.read_csv("test_V2.csv",low_memory=False,delimiter="," )
"""dataset1=pd.read_csv("train_V2.csv",low_memory=False,delimiter="," )"""
X=dataset[:5]
y = dataset[5:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  
clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))
print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))
