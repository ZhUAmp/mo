import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score, davies_bouldin_score




FILE = "[UCI] AAAI-13 Accepted Papers - Papers.csv"
df = pd.read_csv(FILE)

print("Столбцы:", df.columns)


text_column = "Title"
df = df.dropna(subset=[text_column])


df_sample = df.sample(n=min(len(df), 1500), random_state=42)


#TF-IDF+масштабирование

vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
X_tfidf = vectorizer.fit_transform(df_sample[text_column])

scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X_tfidf)

print("Shape TF-IDF:", X_scaled.shape)


experiments = []


def evaluate(name, labels, X):
    """Метрики кластеризации"""
    if len(set(labels)) == 1:
        return None, None

    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X.toarray(), labels)
    return sil, db


#KMEANS
for k in [3, 5, 7, 10]:
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    sil, db = evaluate("KMeans", labels, X_scaled)
    experiments.append(["KMeans", k, None, sil, db])

#Agglomerative
for k in [3, 5, 7]:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X_scaled.toarray())
    sil, db = evaluate("Agglomerative", labels, X_scaled)
    experiments.append(["Agglomerative", k, None, sil, db])

#DBSCAN
for eps in [0.3, 0.5, 0.7, 1.0]:
    model = DBSCAN(eps=eps, min_samples=5)
    labels = model.fit_predict(X_scaled)
    sil, db = evaluate("DBSCAN", labels, X_scaled)
    experiments.append(["DBSCAN", None, eps, sil, db])

#GMM
for k in [3, 5, 7]:
    model = GaussianMixture(n_components=k, random_state=42)
    labels = model.fit_predict(X_scaled.toarray())
    sil, db = evaluate("GMM", labels, X_scaled)
    experiments.append(["GMM", k, None, sil, db])



results = pd.DataFrame(experiments, columns=["Model", "n_clusters", "eps", "Silhouette", "Davies-Bouldin"])
print("\nРезультаты кластеризации:")
print(results.sort_values("Silhouette", ascending=False))



best = results.sort_values("Silhouette", ascending=False).head(3)
print("\nТоп-3 конфигурации:\n", best)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled.toarray())

plt.figure(figsize=(14, 5))

for i, row in best.iterrows():
    model_name = row["Model"]
    k = row["n_clusters"]
    eps = row["eps"]


    if model_name == "KMeans":
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_scaled)


    elif model_name == "Agglomerative":
        print(f"k ={k}")
        #model = AgglomerativeClustering(n_clusters=k)
        model = AgglomerativeClustering(n_clusters=int(k))
        labels = model.fit_predict(X_scaled.toarray())

    elif model_name == "DBSCAN":
        model = DBSCAN(eps=eps)
        labels = model.fit_predict(X_scaled)

    elif model_name == "GMM":
        model = GaussianMixture(n_components=k, random_state=42)
        labels = model.fit_predict(X_scaled.toarray())


    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=20)
    plt.title(f"{model_name}, clusters={k if k else eps}")
    plt.show()
