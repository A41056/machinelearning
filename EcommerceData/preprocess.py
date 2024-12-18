import pandas as pd

def preprocess_data(df):
    print(f"Initial data shape: {df.shape}")

    # Chuyển đổi cột InvoiceDate thành datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Loại bỏ các đơn hàng trả lại (InvoiceNo bắt đầu bằng 'C')
    df = df[~df['InvoiceNo'].str.startswith('C', na=False)]
    print(f"After removing returns: {df.shape}")

    # Tính tổng giá trị đơn hàng (Quantity * UnitPrice)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # Loại bỏ các hàng có giá trị Quantity hoặc UnitPrice <= 0
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    print(f"After removing invalid Quantity or UnitPrice: {df.shape}")

    # Kiểm tra giá trị thiếu trong CustomerID
    missing_customer_ids = df['CustomerID'].isna().sum()
    print(f"Missing CustomerID values: {missing_customer_ids}")

    # Điền giá trị thiếu trong CustomerID bằng giá trị mặc định (ví dụ: -1)
    df['CustomerID'].fillna(-1, inplace=True)
    print(f"After filling missing CustomerID: {df.shape}")

    # Tổng hợp dữ liệu theo khách hàng
    df_grouped = df.groupby('CustomerID').agg({
        'Quantity': 'sum',
        'TotalPrice': 'sum'
    }).reset_index()
    print(f"After grouping by CustomerID: {df_grouped.shape}")
    
    return df_grouped