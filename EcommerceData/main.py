import pandas as pd
from preprocess import preprocess_data
from eda import eda
from clustering import apply_kmeans, analyze_clusters
from visualization import visualize_clusters

def main():
    # 1. Đọc dữ liệu
    df = pd.read_csv('./EcommerceData/ecommerce-data.csv', encoding='ISO-8859-1')

    # 2. Tiền xử lý dữ liệu
    print("Tiến hành tiền xử lý dữ liệu...")
    df_cleaned = preprocess_data(df)
    print(f"Dữ liệu sau khi tiền xử lý: {df_cleaned.shape[0]} dòng và {df_cleaned.shape[1]} cột.")
    
    # 3. Phân tích dữ liệu (EDA)
    print("Thực hiện phân tích dữ liệu (EDA)...")
    eda(df_cleaned)
    
    # 4. Phân cụm khách hàng với K-Means
    print("Tiến hành phân cụm với K-Means...")
    df_clustered, kmeans_model = apply_kmeans(df_cleaned, n_clusters=3)  # Bạn có thể thay đổi số cụm tại đây
    print("Phân cụm hoàn tất.")
    
    # 5. Trực quan hóa kết quả phân cụm
    print("Trực quan hóa kết quả phân cụm...")
    visualize_clusters(df_clustered)
    
    # 6. Phân tích các cụm
    print("Phân tích các cụm...")
    analyze_clusters(df_clustered)

if __name__ == '__main__':
    main()
