import matplotlib.pyplot as plt
import seaborn as sns

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