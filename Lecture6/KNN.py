

#KNN classifier

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()

X = iris.data 
y = iris.target
names = iris.feature_names


from sklearn import preprocessing

X=preprocessing.StandardScaler().fit(X).transform(X.astype(float))
                                                  
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test=train_test_split(X, y, 
                                test_size=0.2, random_state=4)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
k=20
neigh=KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
Pred_y=neigh.predict(X_test)
print("Accuracy when K=20 is", metrics.accuracy_score(y_test, Pred_y))

error_rate=[]
for i in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i !=y_test))

print(error_rate)

plt.figure(figsize=(10,6))
plt.plot(range(1,21), error_rate, color='blue', linestyle='dashed',
        marker='o', markerfacecolor='red', markersize=10)


plt.title('Error rate vs K value')
plt.xlabel(' K')
plt.ylabel('Error rate')

print("minimum error: ", min(error_rate), "at K=", 
      error_rate.index(min(error_rate))+1)
    


acc=[]
for i in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i=knn.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, pred_i))

print(error_rate)

plt.figure(figsize=(10,6))
plt.plot(range(1,21), acc, color='blue', linestyle='dashed',
        marker='o', markerfacecolor='red', markersize=10)


plt.title('Accuracy vs K value')
plt.xlabel(' K')
plt.ylabel('Accuracy')

print("Maximum accuracy: ", max(acc), "at K=", 
      acc.index(max(acc))+1)



"""

#KNN REGRESSOR

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

np.random.seed(0)
X=np.sort(5*np.random.rand(40,1),axis=0)
T=np.linspace(0,5,500)[:,np.newaxis]
y=np.sin(X).ravel()

y[::5]+=(0.5-np.random.rand(8))


n_neighbors=5

for i, weights in enumerate(["uniform", "distance"]):
    knn=neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_p=knn.fit(X,y).predict(T)
    plt.subplot(2,1,i+1)
    plt.scatter(X, y, color="darkorange", label="data")
    plt.plot(T,y_p,color="navy", label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNeighborsRegressor(k=%i, weights='%s')"%(n_neighbors,weights))

plt.tight_layout()
plt.show()

"""
