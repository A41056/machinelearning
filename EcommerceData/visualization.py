import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

def visualize_clusters(df, n_components=2):
    # Giảm chiều dữ liệu xuống 2D cho trực quan hóa
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(df[['Quantity', 'TotalPrice']])
    
    df_pca = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
    df_pca['Cluster'] = df['Cluster']

    # Vẽ biểu đồ các cụm
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Cluster', palette='viridis')
    plt.title('Customer Segmentation')
    plt.show()