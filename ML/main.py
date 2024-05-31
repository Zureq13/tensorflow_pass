import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

names = pd.read_csv('names.csv', sep=";")

X = names.drop(columns=['isOld'])
y = names['isOld']

# model = DecisionTreeClassifier()
model = joblib.load('mymodel.joblib')
model.fit(X,y,epochs=200)
# joblib.dump(model,'mymodel.joblib')

try:
    age = int(input("Age (number): "))
except ValueError:
    print("Age must be a number")

prediction = model.predict([[age]])
print(prediction)
