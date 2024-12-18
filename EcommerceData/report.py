def generate_report(df_kmeans, df_dbscan, silhouette_kmeans):
    print("\n--- Báo cáo Phân Cụm ---")
    print(f"Silhouette Score K-Means: {silhouette_kmeans:.2f}")
    print(f"Số lượng cụm K-Means: {df_kmeans['Cluster'].nunique()}")
    print(f"Số lượng cụm DBSCAN: {df_dbscan['Cluster'].nunique()}")
    print("Cụm K-Means trung bình:")
    print(df_kmeans.groupby('Cluster')[['Quantity', 'TotalPrice']].mean())
    print("\nCụm DBSCAN trung bình:")
    print(df_dbscan.groupby('Cluster')[['Quantity', 'TotalPrice']].mean())
