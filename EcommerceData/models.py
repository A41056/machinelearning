from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, mean_squared_error

# KNN
def apply_knn(X_train, X_test, y_train, y_test, return_predictions=False):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    if return_predictions:
        return y_pred
    return classification_report(y_test, y_pred)

# Linear Regression
def apply_linear_regression(X_train, X_test, y_train, y_test, return_predictions=False):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    if return_predictions:
        return y_pred
    mse = mean_squared_error(y_test, y_pred)
    return mse

# SVM
def apply_svm(X_train, X_test, y_train, y_test, return_predictions=False):
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    if return_predictions:
        return y_pred
    return classification_report(y_test, y_pred)

# Decision Tree
def apply_decision_tree(X_train, X_test, y_train, y_test, return_predictions=False):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    if return_predictions:
        return y_pred
    return classification_report(y_test, y_pred)