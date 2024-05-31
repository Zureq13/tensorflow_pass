import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score

data = pd.read_csv('username.csv', sep=";")
X = pd.get_dummies(data.drop(columns=['Strong','Identifier']))
y = data['Strong']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8)
X_train = tf.convert_to_tensor(X_train, dtype = tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype = tf.float32)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=50, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train,y_train,epochs=700)

# model = tf.keras.models.load_model('MyModel.keras')

Y = model.predict(X_test)
print(X_test)
Y = [0 if val <0.5 else 1 for val in Y]
acc = accuracy_score(y_test,Y)
print(acc)

if(acc > 0.7):
    model.save('MyModel.keras')
else:
    print("Accuracy to low")