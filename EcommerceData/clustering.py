from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

def apply_kmeans(df, n_clusters=3):
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features = scaler.fit_transform(df[['Quantity', 'TotalPrice']])
    
    # Áp dụng K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features)
    
    return df, kmeans

def apply_dbscan(df, eps=0.5, min_samples=5):
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features = scaler.fit_transform(df[['Quantity', 'TotalPrice']])
    
    # Áp dụng DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = dbscan.fit_predict(features)
    
    return df, dbscan