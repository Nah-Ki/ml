from sklearn.datasets import load_iris
iris = load_iris()
print("Feature Names:", iris.feature_names, "Iris Data:", iris.data, "Target Names:", iris.target_names, "Target:", iris.target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size = .25)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

print("Accuracy=", clf.score(X_test, y_test))

print("Predicted Data")
print(clf.predict(X_test))
prediction = clf.predict(X_test)
print("Test data :")
print(y_test)

diff = prediction - y_test
print("Result is ")
print(diff)
print('Total no of samples misclassified =', sum(abs(diff)))
