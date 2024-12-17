import pandas as pd

def preprocess_data(df):
    # Chuyển đổi cột InvoiceDate thành datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Loại bỏ các đơn hàng trả lại (InvoiceNo bắt đầu bằng 'C')
    df = df[df['InvoiceNo'].str.startswith('C') == False]

    # Tính tổng giá trị đơn hàng (Quantity * UnitPrice)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Loại bỏ các hàng có giá trị 'Quantity' hoặc 'UnitPrice' <= 0
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    # Xử lý các giá trị thiếu (nếu có)
    df = df.dropna(subset=['CustomerID'])

    return df