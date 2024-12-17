import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class Preprocessing:
    def __init__(self, df):
        self.df = df
    
    def clean_column_names(self):
        # Loại bỏ khoảng trắng và ký tự đặc biệt trong tên cột
        self.df.columns = self.df.columns.str.strip().str.replace(r'[^\w\s]', '', regex=True)
    
    def handle_missing_values(self):
        # Sử dụng SimpleImputer để thay thế giá trị thiếu bằng giá trị trung bình hoặc mode tùy theo từng cột
        imputer = SimpleImputer(strategy='most_frequent')
        self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)
        return self.df
    
    def encode_categorical_features(self):
        # Mã hóa các cột phân loại
        label_encoder = LabelEncoder()
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            self.df[column] = label_encoder.fit_transform(self.df[column])
        
        return self.df
    
    def scale_features(self):
        # Chuẩn hóa dữ liệu (đặc biệt là các cột số)
        scaler = StandardScaler()
        numerical_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        self.df[numerical_columns] = scaler.fit_transform(self.df[numerical_columns])
        return self.df
    
    def preprocess(self):
        # Làm sạch tên cột trước khi tiền xử lý
        self.clean_column_names()
        # Tiến hành xử lý các bước còn lại
        self.df = self.handle_missing_values()
        self.df = self.encode_categorical_features()
        self.df = self.scale_features()
        return self.df