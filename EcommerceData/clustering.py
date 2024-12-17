from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def apply_kmeans(df, n_clusters=3):
    # Lấy các đặc trưng quan trọng cho phân cụm
    features = df[['Quantity', 'TotalPrice']].copy()
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Áp dụng KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features_scaled)
    
    return df, kmeans

def analyze_clusters(df):
    # Phân tích trung bình của các cụm
    cluster_summary = df.groupby('Cluster').agg(
        avg_quantity=('Quantity', 'mean'),
        avg_total_price=('TotalPrice', 'mean')
    )
    print(cluster_summary)