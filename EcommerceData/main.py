import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, silhouette_score
from clustering import apply_kmeans, apply_classifiers, apply_voting_classifier
from preprocess import preprocess_data
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    # Thiết lập biến môi trường LOKY_MAX_CPU_COUNT
    os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Thay đổi số 4 thành số lượng lõi bạn muốn sử dụng

    # 1. Đọc và tiền xử lý dữ liệu
    print("Đang đọc dữ liệu và tiền xử lý...")
    df_cleaned = preprocess_data('./EcommerceData/ecommerce-data.csv')
    print(f"Dữ liệu sau tiền xử lý: {df_cleaned.shape[0]} dòng và {df_cleaned.shape[1]} cột.")
    
    # 2. Áp dụng K-Means
    print("Phân cụm khách hàng bằng K-Means...")
    df_kmeans, kmeans_model = apply_kmeans(df_cleaned, n_clusters=3)
    
    # 3. Chuẩn hóa và tính Silhouette Score cho K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cleaned[['Quantity', 'TotalPrice']])
    silhouette_kmeans = silhouette_score(X_scaled, df_kmeans['Cluster'])
    print(f"Silhouette Score cho K-Means: {silhouette_kmeans:.2f}")
    
    # 4. Phân chia dữ liệu train/test
    print("Chia dữ liệu cho các thuật toán phân loại và dự đoán...")
    X = df_cleaned[['Quantity', 'TotalPrice']]
    y_classification = df_cleaned['Cluster']  # Sử dụng cụm làm mục tiêu phân loại
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42)
    
    # 5. Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 6. Áp dụng các thuật toán phân loại
    print("Áp dụng các thuật toán phân loại...")
    results_classification = apply_classifiers(X_train, X_test, y_train_class, y_test_class)
    
    # 7. Áp dụng Voting Classifier
    print("Áp dụng Voting Classifier...")
    voting_precision = apply_voting_classifier(X_train, X_test, y_train_class, y_test_class)
    results_classification["Voting Classifier"] = voting_precision
    
    # 8. In kết quả
    print("Kết quả phân loại:")
    for name, precision in results_classification.items():
        print(f"{name} Precision: {precision:.2f} %")

if __name__ == '__main__':
    main()