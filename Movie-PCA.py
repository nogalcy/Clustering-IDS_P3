import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
import seaborn as sns


rating_data = pd.read_csv("csv-metadata/ratings.csv")
movies_data = pd.read_csv("csv-metadata/movies.csv")
user_movie_matrix = rating_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

top_movies_by_users = rating_data['movieId'].value_counts().head(3)

top_movies_info = pd.merge(top_movies_by_users, movies_data, left_index=True, right_on='movieId')
top_movies_info = top_movies_info[['title', 'movieId', 'count']]
top_movies_info.columns = ['Title', 'MovieID', 'Number of Ratings']

print("Top 3 movies by number of users:")
print(top_movies_info)

top_users_by_ratings = rating_data['userId'].value_counts().head(3)

print("\nTop 3 users by number of ratings:")
print(top_users_by_ratings)

k_values = [2, 4, 8, 16, 32, 64, 128]
inertia_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(user_movie_matrix)
    inertia_scores.append(kmeans.inertia_)

plt.plot(k_values, inertia_scores, marker="o")
plt.title("Plot of Inertia Scores for K Values")
plt.xlabel("K value (Number of Clusters)")
plt.ylabel("Inertia Score")
plt.show()

# To me, it appears that the elbow of the plot is at the 32 cluster mark

chosen_k = 32
kmeans_chosen = KMeans(n_clusters=chosen_k)
user_movie_matrix["cluster"] = kmeans_chosen.fit_predict(user_movie_matrix)

top_movies_per_cluster = pd.DataFrame()

for cluster in range(chosen_k):
    cluster_indices = user_movie_matrix[user_movie_matrix["cluster"] == cluster].index
    cluster_ratings = user_movie_matrix.loc[cluster_indices].mean(axis=0).drop(["cluster"])

    # Convert indices to movie IDs
    top_movies = cluster_ratings.sort_values(ascending=False).head(3)
    top_movies_ids = top_movies.index

    top_movies_per_cluster[f"Cluster {cluster + 1}"] = top_movies_ids

# Display movie titles for each cluster
print("Top 3 Movies by Cluster:")
for cluster in range(chosen_k):
    print(f"Cluster {cluster + 1} - Top 3 Movies:")
    for movie_id in top_movies_per_cluster[f"Cluster {cluster + 1}"]:
        title = movies_data.loc[movies_data['movieId'] == movie_id, 'title'].values[0]
        print(f"  {title}")
    print()

user_movie_transposed = user_movie_matrix.T
user_movie_transposed = user_movie_transposed.fillna(0).drop("cluster")

mean_centered = user_movie_transposed - user_movie_transposed.mean(axis=0)
mean_centered = mean_centered.rename(str, axis="columns")

pca = PCA(n_components=2)
pca_result = pca.fit_transform(mean_centered)

movies_data['first_genre'] = movies_data['genres'].str.split('|').str[0]

# Calculate unique color codes for the extracted genres
genre_colors = pd.Categorical(movies_data['first_genre'], categories=sorted(set(movies_data['first_genre']))).codes

random_indices = np.random.choice(len(pca_result), size=4000, replace=False)
pca_result_sampled = pca_result[random_indices]
genre_colors_sampled = genre_colors[random_indices]

# Create a more distinguishable color palette
palette = sns.color_palette("tab10", n_colors=len(set(movies_data['first_genre'])))
palette_list = list(palette)
# Plot the random sample of PCA points with unique and distinguishable colors
scatter = plt.scatter(pca_result_sampled[:, 0], pca_result_sampled[:, 1], c=[palette_list[i] for i in genre_colors_sampled])

plt.title("Random Sample of PCA Movie Results")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

print("Percent of variance explained by 2 components:", sum(pca.explained_variance_ratio_))
'''Honestly, the main pattern I see is that it appears there is not nearly enough dimensions to explain this data. There is 
a massive clump of points at the 0-0 mark area and it appears that is where most of the data points reside, meaning we probably are not explaining much of the variance 
'''

pca = PCA(n_components=0.8)
pca_80 = pca.fit(mean_centered)
n_components_80 = pca_80.n_components_

# Determine the number of components needed to explain 40% of the variance
pca = PCA(n_components=0.4)
pca_40 = pca.fit(mean_centered)
n_components_40 = pca_40.n_components_

print("Number of components for 80%:", n_components_80)
print("Number of components for 40%:", n_components_40)

user_movie_matrix_2 = user_movie_matrix.drop("cluster", axis=1)
user_movie_matrix_2.columns = user_movie_matrix_2.columns.astype(str)

svd = TruncatedSVD(n_components=128)
svd.fit(user_movie_matrix_2)
plt.scatter(range(1,129),svd.singular_values_, marker = "o")
plt.title("Singular Values for k=128")
plt.xlabel("Component Number")
plt.ylabel("Singular Value")
plt.show()

explained_variance_ratios = []
for k in [2,4,8,16,32,64,128]:
    svd = TruncatedSVD(n_components=k)
    svd.fit(user_movie_matrix_2)
    explained_ratio = np.sum(svd.explained_variance_ratio_)
    explained_variance_ratios.append(explained_ratio)
    print(f"Explained Variance Ratio for k={k}: {explained_ratio:.4f}")

svd_2 = TruncatedSVD(n_components=2)
transformed_svd2 = svd_2.fit_transform(user_movie_matrix_2)
plt.scatter(transformed_svd2[:,0], transformed_svd2[:,1], c=user_movie_matrix["cluster"], cmap="viridis")
plt.title("SVD Results (k = 2) with User Clusters")
plt.xlabel("SVD 1")
plt.ylabel("SVD 2")
plt.show()


