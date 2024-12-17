import pandas as pd
from preprocess import Preprocessing
from eda import EDA
from clustering import Clustering
from visualization import Visualization

def main():
    # Đọc dữ liệu
    df = pd.read_csv('./EmployeeAttritionPrediction/employee-attrition-prediction.csv', encoding='utf-8-sig')
    
    # Tiền xử lý dữ liệu
    preprocessing = Preprocessing(df)
    df_cleaned = preprocessing.preprocess()
    print(df_cleaned.columns)
    print(f"Dữ liệu sau khi tiền xử lý: {df_cleaned.shape[0]} dòng và {df_cleaned.shape[1]} cột.")
    
    # Phân tích dữ liệu (EDA)
    eda = EDA(df_cleaned)
    eda.perform_eda()
    
    # Phân cụm với KMeans
    clustering = Clustering(df_cleaned)
    df_clustered, kmeans_model = clustering.perform_clustering(n_clusters=3)
    
    # Trực quan hóa kết quả phân cụm
    visualization = Visualization(df_clustered)
    visualization.visualize_clusters()

if __name__ == '__main__':
    main()