from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score, silhouette_score
from models import apply_knn, apply_linear_regression, apply_svm, apply_decision_tree
from clustering import apply_kmeans, apply_dbscan
from preprocess import preprocess_data
from report import generate_report
from visualization import visualize_classification_results, visualize_regression_results, compare_algorithms, visualize_cluster_heatmap, visualize_correlation_matrix
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():

    # 1. Đọc dữ liệu
    print("Đang đọc dữ liệu...")
    df = pd.read_csv('./EcommerceData/ecommerce-data.csv', encoding='ISO-8859-1')
    
    # 2. Tiền xử lý dữ liệu
    print("Tiến hành tiền xử lý dữ liệu...")
    df_cleaned = preprocess_data(df)
    print(f"Dữ liệu sau tiền xử lý: {df_cleaned.shape[0]} dòng và {df_cleaned.shape[1]} cột.")
    
    # 3. Áp dụng K-Means
    print("Phân cụm khách hàng bằng K-Means...")
    df_kmeans, kmeans_model = apply_kmeans(df_cleaned, n_clusters=3)
    
    # 4. Áp dụng DBSCAN
    print("Phân cụm khách hàng bằng DBSCAN...")
    df_dbscan, dbscan_model = apply_dbscan(df_cleaned)
    
    # 5. Chuẩn hóa và tính Silhouette Score cho K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cleaned[['Quantity', 'TotalPrice']])
    silhouette_kmeans = silhouette_score(X_scaled, df_kmeans['Cluster'])

    # 6. Phân chia dữ liệu train/test
    print("Chia dữ liệu cho các thuật toán phân loại và dự đoán...")
    X = df_cleaned[['Quantity', 'TotalPrice']]
    y_classification = (df_cleaned['TotalPrice'] > df_cleaned['TotalPrice'].mean()).astype(int)  # Mục tiêu phân loại
    y_regression = df_cleaned['TotalPrice']  # Mục tiêu hồi quy
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42)
    _, _, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
    
    # 7. Áp dụng các thuật toán
    print("Áp dụng KNN...")
    y_pred_knn = apply_knn(X_train, X_test, y_train_class, y_test_class, return_predictions=True)
    knn_report = classification_report(y_test_class, y_pred_knn)
    print(knn_report)

    print("Áp dụng Linear Regression...")
    y_pred_lr = apply_linear_regression(X_train, X_test, y_train_reg, y_test_reg, return_predictions=True)
    lr_mse = mean_squared_error(y_test_reg, y_pred_lr)
    print(f"Linear Regression Mean Squared Error: {lr_mse:.2f}")
    
    print("Áp dụng SVM...")
    y_pred_svm = apply_svm(X_train, X_test, y_train_class, y_test_class, return_predictions=True)
    svm_report = classification_report(y_test_class, y_pred_svm)
    print(svm_report)
    
    print("Áp dụng Decision Tree...")
    y_pred_dt = apply_decision_tree(X_train, X_test, y_train_class, y_test_class, return_predictions=True)
    dt_report = classification_report(y_test_class, y_pred_dt)
    print(dt_report)
    
    # 8. Tạo báo cáo
    print("Tạo báo cáo kết quả...")
    generate_report(df_kmeans, df_dbscan, silhouette_kmeans)
    print("Hoàn thành!")

    # 9. Visualization
    print("Visualizing KNN results...")
    visualize_classification_results(y_test_class, y_pred_knn, method_name="KNN")

    print("Visualizing SVM results...")
    visualize_classification_results(y_test_class, y_pred_svm, method_name="SVM")

    print("Visualizing Decision Tree results...")
    visualize_classification_results(y_test_class, y_pred_dt, method_name="Decision Tree")

    print("Visualizing Linear Regression results...")
    visualize_regression_results(y_test_reg, y_pred_lr, method_name="Linear Regression")

    # 10. So sánh hiệu suất các thuật toán
    results_classification = {
        "KNN": accuracy_score(y_test_class, y_pred_knn),
        "SVM": accuracy_score(y_test_class, y_pred_svm),
        "Decision Tree": accuracy_score(y_test_class, y_pred_dt),
    }
    results_regression = {"Linear Regression (MSE)": lr_mse}
    compare_algorithms(results_classification, results_regression)

    # Vẽ Heatmap cho K-Means
    print("Visualizing cluster heatmap for K-Means...")
    visualize_cluster_heatmap(df_kmeans, clusters_col='Cluster', features=['Quantity', 'TotalPrice'])

    # Vẽ Heatmap cho DBSCAN
    print("Visualizing cluster heatmap for DBSCAN...")
    visualize_cluster_heatmap(df_dbscan, clusters_col='Cluster', features=['Quantity', 'TotalPrice'])

    # Vẽ ma trận tương quan
    print("Visualizing correlation matrix...")
    visualize_correlation_matrix(df_cleaned)
if __name__ == '__main__':
    main()