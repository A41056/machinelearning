import pandas as pd

def preprocess_data(filepath):
    # Đọc dữ liệu
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    print(f"Initial data shape: {df.shape}")

    # Chuyển đổi cột InvoiceDate thành datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Loại bỏ các hàng có CustomerID rỗng
    df = df.dropna(subset=['CustomerID'])
    print(f"Dataframe dimensions after removing null CustomerID: {df.shape}")
    
    # Loại bỏ các hàng trùng lặp
    df = df.drop_duplicates()
    print(f"Dataframe dimensions after removing duplicates: {df.shape}")

    # Tạo cột mới để đánh dấu các đơn hàng bị hủy
    df['order_canceled'] = df['InvoiceNo'].str.startswith('C').astype(int)
    
    # Đếm số lượng đơn hàng bị hủy
    num_cancellations = df['order_canceled'].sum()
    total_transactions = df['InvoiceNo'].nunique()
    print(f"Number of orders canceled: {num_cancellations}/{total_transactions} ({(num_cancellations / total_transactions) * 100:.2f}%)")
    
    # Xử lý các mã sản phẩm đặc biệt
    special_codes = df[df['StockCode'].str.contains(r'^[A-Z]+$', regex=True)]['StockCode'].unique()
    print("Special Stock Codes:")
    print(special_codes)
    
    # Tính tổng giá trị đơn hàng
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    return df

def save_cleaned_data(df, filepath):
    df.to_csv(filepath, index=False)
    print(f"Cleaned data saved to {filepath}")

# Đường dẫn tới tệp dữ liệu
input_filepath = './EcommerceData/ecommerce-data.csv'
output_filepath = './EcommerceData/cleaned_ecommerce_data.csv'

# Thực hiện tiền xử lý dữ liệu
df_cleaned = preprocess_data(input_filepath)

# Lưu dữ liệu đã chuẩn bị
save_cleaned_data(df_cleaned, output_filepath)