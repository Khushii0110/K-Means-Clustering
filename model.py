from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import os
#code By Khushi Tiwari
if not os.path.exists("Graphs"):
    os.makedirs("Graphs")

################################################################# 1) Data Import and Preprocessing
data = pd.read_csv("userbehaviour.csv")
null_values = data.isnull().sum()
print("Null values in the dataset:\n", null_values)
column_info = data.info()
print("\nColumn information:\n", column_info)
descriptive_stats = data.describe()
print("\nDescriptive statistics:\n", descriptive_stats)

################################################################# 2) Analysis of Screen Time and Spending Capacity
highest_screen_time = data['Average Screen Time'].max()
print("Highest screen time:", highest_screen_time)
lowest_screen_time = data['Average Screen Time'].min()
print("Lowest screen time:", lowest_screen_time)
average_screen_time = data['Average Screen Time'].mean()
print("Average screen time:", average_screen_time)
highest_amount_spent = data['Average Spent on App (INR)'].max()
print("Highest amount spent:", highest_amount_spent)
lowest_amount_spent = data['Average Spent on App (INR)'].min()
print("Lowest amount spent:", lowest_amount_spent)
average_amount_spent = data['Average Spent on App (INR)'].mean()
print("Average amount spent:", average_amount_spent)

################################################################# 3) Relationship Analysis - Active Users vs. Uninstalled Users
installed_users = data[data['Status'] == 'Installed']
uninstalled_users = data[data['Status'] == 'Uninstalled']

plt.figure(figsize=(12, 8))
plt.scatter(installed_users['Average Screen Time'], installed_users['Average Spent on App (INR)'], 
            s=installed_users['Ratings']*10, c='blue', alpha=0.5, label='Installed')
plt.scatter(uninstalled_users['Average Screen Time'], uninstalled_users['Average Spent on App (INR)'], 
            s=uninstalled_users['Ratings']*10, c='red', alpha=0.5, label='Uninstalled')

plt.ylabel('Average Spent on App (INR)')
plt.xlabel('Average Screen Time')
plt.title('Spending Capacity vs Screen Time')
plt.legend()
plt.grid(True)
plt.savefig('Graphs/Spending Capacity vs Screen Time.png')
plt.show()

################################################################# 4) Relationship Analysis - Ratings vs. Screen Time
installed_apps = data[data['Status'] == 'Installed']
uninstalled_apps = data[data['Status'] == 'Uninstalled']

plt.figure(figsize=(12, 8))
plt.scatter(installed_apps['Average Screen Time'], installed_apps['Ratings'], 
            s=installed_apps['Average Spent on App (INR)']*.4, c='blue', alpha=0.5, label='Installed Apps')
plt.scatter(uninstalled_apps['Average Screen Time'], uninstalled_apps['Ratings'], 
            s=uninstalled_apps['Average Spent on App (INR)']*.4, c='red', alpha=0.5, label='Uninstalled Apps')

plt.xlabel('Average Screen Time')
plt.ylabel('Ratings')
plt.title('Ratings vs Screen Time Bubble Graph')
plt.legend()
plt.grid(True)
plt.savefig('Graphs/Ratings vs Screen Time.png')
plt.show()

################################################################# 5) User Segmentation with K-means Clustering
label_encoder = LabelEncoder()
data['Status'] = label_encoder.fit_transform(data['Status'])
kmeans = KMeans(n_clusters=3, random_state=0).fit(data.drop(['userid'], axis=1))
data['Cluster'] = kmeans.labels_

colors = {0: 'blue', 1: 'green', 2: 'red'}
cluster_labels = {0: 'Retained', 1: 'Churn', 2: 'Needs Attention'}
plt.figure(figsize=(12, 8))
for cluster_num, color in colors.items():
    cluster_data = data[data['Cluster'] == cluster_num]
    plt.scatter(cluster_data['Last Visited Minutes'], cluster_data['Average Spent on App (INR)'], c=color, label=cluster_labels[cluster_num], s=50)

plt.ylabel('Average Spent on App (INR)')
plt.xlabel('Last Visited Minutes')
plt.title('User Segmentation with K-means Clustering')
plt.legend()
plt.savefig('Graphs/User Segmentation with K-means Clustering.png')
plt.show()

