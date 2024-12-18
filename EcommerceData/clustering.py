from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, precision_score

def apply_kmeans(df, n_clusters=3):
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features = scaler.fit_transform(df[['Quantity', 'TotalPrice']])
    
    # Áp dụng K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features)
    
    return df, kmeans

def classify_customers(X_train, X_test, y_train, y_test, model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    print(f"{model_name} Precision: {precision:.2f} %")
    print(classification_report(y_test, y_pred))
    return precision

def apply_classifiers(X_train, X_test, y_train, y_test):
    classifiers = {
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in classifiers.items():
        precision = classify_customers(X_train, X_test, y_train, y_test, model, name)
        results[name] = precision
    
    return results

def apply_voting_classifier(X_train, X_test, y_train, y_test):
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ],
        voting='hard'
    )
    
    precision = classify_customers(X_train, X_test, y_train, y_test, voting_clf, "Voting Classifier")
    return precision