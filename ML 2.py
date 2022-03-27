from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


iris=datasets.load_iris()


features = iris.data
lables = iris.target


clf =  KNeighborsClassifier()
clf.fit(features,lables)

pdct = clf.predict([[3,8.5,5,6]])
print(pdct)