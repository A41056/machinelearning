import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

class Visualization:
    def __init__(self, df):
        self.df = df
    
    def visualize_clusters(self):
        # Giảm chiều dữ liệu xuống 2D cho trực quan hóa
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(self.df[['Age', 'MonthlyIncome', 'TotalWorkingYears']])
        
        df_pca = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
        df_pca['Cluster'] = self.df['Cluster']
        
        # Vẽ biểu đồ các cụm
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Cluster', palette='viridis')
        plt.title('Customer Segmentation')
        plt.show()