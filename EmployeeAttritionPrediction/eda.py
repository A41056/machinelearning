import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, df):
        self.df = df
    
    def summary_statistics(self):
        # Thống kê mô tả
        print(self.df.describe())
    
    def plot_histograms(self):
        # Vẽ histogram cho các cột số
        numerical_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()

    def plot_correlation_matrix(self):
        # Vẽ ma trận tương quan
        plt.figure(figsize=(10, 8))
        corr = self.df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()
    
    def plot_category_distribution(self):
        # Vẽ phân phối cho các cột phân loại (vd: 'Attrition', 'Gender')
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            plt.figure(figsize=(8, 6))
            sns.countplot(x=col, data=self.df)
            plt.title(f'Distribution of {col}')
            plt.show()
    
    def perform_eda(self):
        self.summary_statistics()
        self.plot_histograms()
        self.plot_correlation_matrix()
        self.plot_category_distribution()
