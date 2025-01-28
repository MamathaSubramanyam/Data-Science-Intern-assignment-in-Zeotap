import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
# Load data from CSV files (downloaded locally for simplicity)
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Inspect data
print(customers.head())
print(products.head())
print(transactions.head())
# Convert date columns to datetime
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Check for missing values
print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())

# Merge transactions with products and customers
transactions_full = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")

# 1. Analyze customer sign-ups by region
signup_by_region = customers['Region'].value_counts()
signup_by_region.plot(kind='bar', title='Signups by Region')
plt.show()

# 2. Sales trend over time
sales_trend = transactions.groupby(transactions['TransactionDate'].dt.to_period('M')).sum()['TotalValue']
sales_trend.plot(title='Sales Trend Over Time')
plt.show()

# 3. Most popular product categories
popular_categories = transactions_full['Category'].value_counts()
popular_categories.plot(kind='bar', title='Popular Product Categories')
plt.show()

# 4. Customers with the highest transaction values
top_customers = transactions.groupby('CustomerID').sum()['TotalValue'].sort_values(ascending=False).head(10)
top_customers.plot(kind='bar', title='Top Customers by Total Spend')
plt.show()

# 5. Average transaction value per region
avg_transaction_region = transactions_full.groupby('Region').mean()['TotalValue']
avg_transaction_region.plot(kind='bar', title='Average Transaction Value by Region')
plt.show()