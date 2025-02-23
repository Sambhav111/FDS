import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:\\Users\\SAMBHAV SHARMA\\OneDrive\\Desktop\\customer_data1.csv"  # Ensure this file is in the working directory
df = pd.read_csv(file_path)

# Basic Probability Calculation
product_counts = df['Product_Category'].value_counts()
total_transactions = len(df)
product_probabilities = product_counts / total_transactions
print("Basic Probability Calculation:")
print(product_probabilities)

# Expected Purchase Amount
expected_value = (df['Purchase_Amount'].sum()) / total_transactions
print("\nExpected Purchase Amount (E[X]):", expected_value)

# Joint Probability (Product & Payment Method)
joint_prob_table = pd.crosstab(df['Product_Category'], df['Payment_Method'], normalize=True)
print("\nJoint Probability Table:")
print(joint_prob_table)

# Conditional Probability P(Payment Method | Product Category)
conditional_prob_table = pd.crosstab(df['Product_Category'], df['Payment_Method'], normalize='index')
print("\nConditional Probability Table:")
print(conditional_prob_table)

# Monthly Sales Trend
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], format="%d-%m-%Y")
df['Month_Year'] = df['Purchase_Date'].dt.to_period('M')
monthly_sales = df.groupby('Month_Year')['Purchase_Amount'].sum()
plt.figure(figsize=(10,5))
monthly_sales.plot(kind='bar', color='skyblue')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Purchase Amount')
plt.xticks(rotation=45)
plt.show()

# Spending Pattern Distribution
sns.boxplot(y=df['Purchase_Amount'])
plt.title('Spending Pattern (Box Plot)')
plt.show()

# Joint Probability Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(joint_prob_table, annot=True, cmap="Blues")
plt.title("Joint Probability Heatmap")
plt.show()
