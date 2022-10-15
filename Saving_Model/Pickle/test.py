import pickle

#laod the model

model = pickle.load('diabetes_79.pkl')

result = model.predict([[1,1,1,1,1,1,1,1]])

if result[0]==1:
    print("diabetic")
else:
    print('not diabetic')