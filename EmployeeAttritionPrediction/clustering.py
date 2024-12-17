from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class Clustering:
    def __init__(self, df):
        self.df = df
    
    def apply_kmeans(self, n_clusters=3):
        # Lấy các đặc trưng quan trọng cho phân cụm
        features = self.df[['Age', 'MonthlyIncome', 'TotalWorkingYears']].copy()
        
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Áp dụng KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(features_scaled)
        
        return self.df, kmeans

    def analyze_clusters(self):
        # Phân tích trung bình của các cụm
        cluster_summary = self.df.groupby('Cluster').agg(
            avg_age=('Age', 'mean'),
            avg_income=('MonthlyIncome', 'mean'),
            avg_working_years=('TotalWorkingYears', 'mean')
        )
        print(cluster_summary)
    
    def perform_clustering(self, n_clusters=3):
        self.df, kmeans_model = self.apply_kmeans(n_clusters)
        self.analyze_clusters()
        return self.df, kmeans_model