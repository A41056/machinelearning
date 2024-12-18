import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def visualize_classification_results(y_test, y_pred, method_name):
    """Biểu đồ confusion matrix trực quan hóa kết quả phân loại."""
    plt.figure(figsize=(8, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title(f"Confusion Matrix - {method_name}")
    plt.show()
    plt.close('all')

def visualize_regression_results(y_test, y_pred, method_name):
    """Biểu đồ trực quan hóa kết quả hồi quy."""
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"Regression Results - {method_name}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()
    plt.close('all')

def compare_algorithms(results_classification, results_regression):
    """So sánh hiệu suất các thuật toán."""
    # So sánh các thuật toán phân loại
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results_classification.keys()), y=list(results_classification.values()))
    plt.title("Comparison of Classification Algorithms")
    plt.xlabel("Algorithm")
    plt.ylabel("Accuracy")
    plt.show()
    plt.close('all')
    
    # So sánh hồi quy
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results_regression.keys()), y=list(results_regression.values()))
    plt.title("Comparison of Regression Algorithms")
    plt.xlabel("Algorithm")
    plt.ylabel("MSE")
    plt.show()
    plt.close('all')

def visualize_cluster_heatmap(df, clusters_col, features):
    """Biểu đồ heatmap cho các cụm."""
    plt.figure(figsize=(10, 8))
    cluster_summary = df.groupby(clusters_col)[features].mean()
    sns.heatmap(cluster_summary, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(f"Cluster Heatmap - {clusters_col}")
    plt.xlabel("Features")
    plt.ylabel("Clusters")
    plt.show()
    plt.close('all')

def visualize_correlation_matrix(df):
    """Biểu đồ ma trận tương quan."""
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
    plt.title("Correlation Matrix")
    plt.show()
    plt.close('all')  # Close all figures