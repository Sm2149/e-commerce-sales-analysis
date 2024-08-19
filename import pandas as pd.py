import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pandas.plotting import table
import xlsxwriter

# Define the path to your CSV file
file_path = 'C:\\Users\\SAIDATTA\\Desktop\\data.csv\\data.csv'

def load_data(file_path):
    """
    Load data from a CSV file with different encodings.
    """
    encodings = ['utf-8', 'latin1', 'cp1252']
    data = None
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully read the file with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            print(f"Failed to read the file with encoding: {encoding}")
        except Exception as e:
            print(f"An error occurred: {e}")
    if data is None:
        raise Exception("Failed to read the CSV file with all tried encodings.")
    return data

def clean_data(data):
    """
    Clean and preprocess the data.
    """
    # Fill missing values
    data['Description'].fillna('Unknown', inplace=True)
    data['CustomerID'].fillna(0, inplace=True)

    # Compute SalesAmount if not present
    if 'SalesAmount' not in data.columns:
        if 'Quantity' in data.columns and 'UnitPrice' in data.columns:
            data['SalesAmount'] = data['Quantity'] * data['UnitPrice']
        else:
            raise Exception("Required columns 'Quantity' or 'UnitPrice' are missing for SalesAmount calculation.")

    # Convert the 'InvoiceDate' column to datetime format if it exists
    if 'InvoiceDate' in data.columns:
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')

        # Extract 'Year', 'Month', and 'Day' from 'InvoiceDate'
        data['Year'] = data['InvoiceDate'].dt.year
        data['Month'] = data['InvoiceDate'].dt.month
        data['Day'] = data['InvoiceDate'].dt.day
    else:
        raise Exception("'InvoiceDate' column is missing.")
    
    return data

def plot_sales_trends(data):
    """
    Plot monthly sales trends.
    """
    if 'Year' in data.columns and 'Month' in data.columns and 'SalesAmount' in data.columns:
        monthly_sales = data.groupby(['Year', 'Month']).agg({'SalesAmount': 'sum'}).reset_index()
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=monthly_sales, x='Month', y='SalesAmount', hue='Year', marker="o")
        plt.title('Monthly Sales Trends')
        plt.xlabel('Month')
        plt.ylabel('Total Sales')
        plt.legend(title='Year')
        plt.grid(True)
        plt.show()
    else:
        print("Required columns for sales trends are missing.")

def plot_customer_behavior(data):
    """
    Plot customer spending distribution.
    """
    if 'CustomerID' in data.columns and 'SalesAmount' in data.columns and 'InvoiceNo' in data.columns:
        customer_behavior = data.groupby('CustomerID').agg({'SalesAmount': 'sum', 'InvoiceNo': 'count'}).reset_index()
        plt.figure(figsize=(12, 6))
        sns.histplot(customer_behavior['SalesAmount'], bins=30, kde=True)
        plt.title('Customer Spending Distribution')
        plt.xlabel('Total Sales')
        plt.ylabel('Number of Customers')
        plt.grid(True)
        plt.show()
    else:
        print("Required columns for customer behavior analysis are missing.")

def plot_product_popularity(data):
    """
    Plot the top 10 products by sales.
    """
    if 'StockCode' in data.columns and 'SalesAmount' in data.columns and 'InvoiceNo' in data.columns:
        product_popularity = data.groupby('StockCode').agg({'SalesAmount': 'sum', 'InvoiceNo': 'count'}).reset_index()
        top_products = product_popularity.sort_values('SalesAmount', ascending=False).head(10)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_products, x='StockCode', y='SalesAmount', palette="viridis")
        plt.title('Top 10 Products by Sales')
        plt.xlabel('Product Code')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.show()
    else:
        print("Required columns for product popularity analysis are missing.")

def plot_customer_segments(data):
    """
    Perform customer segmentation using different clustering algorithms and plot the segments.
    """
    if 'CustomerID' in data.columns and 'SalesAmount' in data.columns and 'InvoiceNo' in data.columns:
        customer_behavior = data.groupby('CustomerID').agg({'SalesAmount': 'sum', 'InvoiceNo': 'count'}).reset_index()
        X = customer_behavior[['SalesAmount', 'InvoiceNo']]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # KMeans Clustering
        kmeans = KMeans(n_clusters=3, random_state=0)
        customer_behavior['KMeans_Segment'] = kmeans.fit_predict(X_scaled)
        kmeans_silhouette = silhouette_score(X_scaled, customer_behavior['KMeans_Segment'])
        print(f"KMeans Silhouette Score: {kmeans_silhouette:.3f}")

        # DBSCAN Clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        customer_behavior['DBSCAN_Segment'] = dbscan.fit_predict(X_scaled)
        dbscan_silhouette = silhouette_score(X_scaled, customer_behavior['DBSCAN_Segment'])
        print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.3f}")

        # Hierarchical Clustering
        agglo = AgglomerativeClustering(n_clusters=3)
        customer_behavior['Agglo_Segment'] = agglo.fit_predict(X_scaled)
        agglo_silhouette = silhouette_score(X_scaled, customer_behavior['Agglo_Segment'])
        print(f"Agglomerative Clustering Silhouette Score: {agglo_silhouette:.3f}")

        # Plot KMeans Segmentation
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='SalesAmount', y='InvoiceNo', hue='KMeans_Segment', data=customer_behavior, palette='deep')
        plt.title('Customer Segmentation using KMeans')
        plt.xlabel('Total Sales')
        plt.ylabel('Number of Orders')
        plt.grid(True)
        plt.show()

        # Plot DBSCAN Segmentation
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='SalesAmount', y='InvoiceNo', hue='DBSCAN_Segment', data=customer_behavior, palette='deep')
        plt.title('Customer Segmentation using DBSCAN')
        plt.xlabel('Total Sales')
        plt.ylabel('Number of Orders')
        plt.grid(True)
        plt.show()

        # Plot Hierarchical Segmentation
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='SalesAmount', y='InvoiceNo', hue='Agglo_Segment', data=customer_behavior, palette='deep')
        plt.title('Customer Segmentation using Agglomerative Clustering')
        plt.xlabel('Total Sales')
        plt.ylabel('Number of Orders')
        plt.grid(True)
        plt.show()
    else:
        print("Required columns for customer segmentation are missing.")

def generate_report(data, file_path='report.xlsx'):
    """
    Generate and save a report including data tables and visualizations.
    """
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        # Create a sheet for sales trends
        monthly_sales = data.groupby(['Year', 'Month']).agg({'SalesAmount': 'sum'}).reset_index()
        monthly_sales.to_excel(writer, sheet_name='Monthly Sales', index=False)

        # Create a sheet for customer behavior
        customer_behavior = data.groupby('CustomerID').agg({'SalesAmount': 'sum', 'InvoiceNo': 'count'}).reset_index()
        customer_behavior.to_excel(writer, sheet_name='Customer Behavior', index=False)

        # Plot and save figures
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=monthly_sales, x='Month', y='SalesAmount', hue='Year', marker="o")
        plt.title('Monthly Sales Trends')
        plt.xlabel('Month')
        plt.ylabel('Total Sales')
        plt.savefig('monthly_sales_trends.png')
        plt.close()

        # Add plots to Excel
        workbook = writer.book
        worksheet = workbook.add_worksheet('Sales Trends Chart')
        worksheet.insert_image('B2', 'monthly_sales_trends.png')

# Main script
data = load_data(file_path)
data = clean_data(data)
plot_sales_trends(data)
plot_customer_behavior(data)
plot_product_popularity(data)
plot_customer_segments(data)
generate_report(data)
