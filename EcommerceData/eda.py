import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def eda(df):
    # Số lượng khách hàng và sản phẩm
    print(f"Unique Customers: {df['CustomerID'].nunique()}")
    print(f"Unique Products: {df['StockCode'].nunique()}")
    
    # Số lượng đơn hàng theo quốc gia
    plt.figure(figsize=(10, 6))
    country_counts = df['Country'].value_counts().head(10)
    country_counts.plot(kind='bar')
    plt.title('Top 10 Countries by Orders')
    plt.ylabel('Number of Orders')
    plt.show()

    # Phân phối tổng giá trị đơn hàng
    plt.figure(figsize=(10, 6))
    sns.histplot(df['TotalPrice'], bins=50, kde=True)
    plt.title('Distribution of Total Price')
    plt.xlabel('Total Price')
    plt.ylabel('Frequency')
    plt.show()

def extract_keywords(df):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Description'])
    keywords = vectorizer.get_feature_names_out()
    return X, keywords

def encode_data(df, X, keywords):
    # Mã hóa một-hot cho các từ khóa
    df_keywords = pd.DataFrame(X.toarray(), columns=keywords)
    
    # Thêm các cột phạm vi giá
    price_bins = [0, 1, 2, 3, 5, 10, df['UnitPrice'].max()]
    price_labels = ['0-1', '1-2', '2-3', '3-5', '5-10', '>10']
    df['PriceRange'] = pd.cut(df['UnitPrice'], bins=price_bins, labels=price_labels)
    df_price_range = pd.get_dummies(df['PriceRange'], prefix='PriceRange')
    
    # Kết hợp các cột mã hóa
    df_encoded = pd.concat([df_keywords, df_price_range], axis=1)
    
    return df_encoded

def cluster_products(df_encoded, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_encoded)
    silhouette_avg = silhouette_score(df_encoded, clusters)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")
    return clusters

def analyze_clusters(df, clusters):
    for cluster in range(clusters.max() + 1):
        cluster_data = df[df['ProductCluster'] == cluster]
        print(f"Cluster {cluster}: {len(cluster_data)} products")
        print(cluster_data['Description'].value_counts().head(10))
        print()

def product_analysis(df):
    # Trích xuất từ khóa từ cột Description
    X, keywords = extract_keywords(df)
    
    # Đếm số lần xuất hiện của mỗi từ khóa
    counts = X.sum(axis=0).A1
    
    # Mã hóa dữ liệu
    df_encoded = encode_data(df, X, keywords)
    
    # Tạo các cụm sản phẩm
    clusters = cluster_products(df_encoded, n_clusters=5)
    df['ProductCluster'] = clusters
    
    # Phân tích các cụm sản phẩm
    analyze_clusters(df, clusters)

# Đường dẫn tới tệp dữ liệu
input_filepath = './EcommerceData/cleaned_ecommerce_data.csv'

# Đọc dữ liệu đã làm sạch
df_cleaned = pd.read_csv(input_filepath)

# Thực hiện EDA
eda(df_cleaned)

# Phân tích sản phẩm
product_analysis(df_cleaned)