# ----------------------------------------------------------------------
#               Phase 4: Clustering and Association Rule Mining
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import collections
from mlxtend.frequent_patterns import apriori, association_rules
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings(action='ignore')

# Read Dataset
df = pd.read_csv('credit_classification_cleaned.csv')
print(df.info())
print(df.head())
print(df.columns)
print('Shape:', df.shape)
X = df.drop('Credit_Score',axis=1)
y = df['Credit_Score']
# Downsample the data

down_sample = RandomUnderSampler(sampling_strategy={0: 4000, 1: 4000, 2: 4000}, random_state=5805)
X, y = down_sample.fit_resample(X, y)
# df = df.sample(18000, random_state=5805)
# X = df.drop('Credit_Score',axis=1)
# Prepare data X
X = X[['Outstanding_Debt','Delay_from_due_date','Changed_Credit_Limit',
        'Credit_Utilization_Ratio','Num_of_Delayed_Payment','Payment_of_Min_Amount_Yes','Credit_Mix_Good']]

# -------------------Clustering Analysis-------------------

# Silhouette Scores
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=5805)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plotting the silhouette scores
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for K-Means Clustering')
plt.show()

# Within cluster sum of squares (WCSS)
wcss = []

# Assume you want to try values of k from 2 to 10
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=5805)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the within-cluster sum of squares
plt.plot(range(2, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares')
plt.title('Within-Cluster Variation Plot for K-Means Clustering')
plt.show()


# # Association Rule Mining
# Prepare Data
# Which occupation is linked to Credit_Mix of good
# Occupations - Scientist,
assoc_df = pd.read_csv('Association.csv')
print(assoc_df.columns)
one_hot_df= pd.get_dummies(data=assoc_df[['Occupation','Credit_Mix','Payment_of_Min_Amount']])
# one_hot_df.drop(['Occupation','Credit_Mix','Payment_of_Min_Amount'],axis=1, inplace=True)
assoc_df = pd.concat([assoc_df,one_hot_df],axis=1)
assoc_df.drop(['Occupation','Credit_Mix','Payment_of_Min_Amount'],axis=1, inplace=True)
print(assoc_df.columns)
print('---------- Results of Association Rule Mining ----------')
df = apriori(assoc_df,min_support=0.2, use_colnames=True, verbose=1)
print(df)
df_ar = association_rules(df,metric='confidence', min_threshold=0.7)
df_ar = df_ar.sort_values(['confidence','lift'], ascending=[False, False])
print(df_ar.to_string())


