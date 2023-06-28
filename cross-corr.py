import pandas as pd
import numpy as np
from itertools import combinations
import networkx as nx
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sys import argv
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")


def standardize_dataframe(df):
    """Standardize the dataframe."""
    return (df - df.mean()) / df.std()


def compute_cross_correlation(df, pair):
    """Compute cross correlation for a pair of columns."""
    xcorrelation = np.correlate(df[pair[0]], df[pair[1]], 'full')
    autocorr_1 = np.correlate(df[pair[0]], df[pair[0]], 'full')
    autocorr_2 = np.correlate(df[pair[1]], df[pair[1]], 'full')
    xcorrelation /= np.sqrt(autocorr_1[len(df[pair[0]])-1] * autocorr_2[len(df[pair[1]])-1])

    lags = np.arange(-len(df[pair[0]]) + 1, len(df[pair[1]]))
    indices = np.where((lags >= -150) & (lags <= 150))

    restricted_lags = lags[indices]
    restricted_xcorrelation = xcorrelation[indices]

    maxlag = restricted_lags[np.argmax(restricted_xcorrelation)]
    maxval = np.max(restricted_xcorrelation)

    return (maxval, -abs(maxlag), pair[0], pair[1])


def choose_pairs(values):
    """Choose pairs based on their values."""
    chosen_pairs=[]
    for val in values:
        # if abs(val[0])>0.1:
        chosen_pairs.append([-abs(val[0]), val[2], val[3]])
    # chosen_pairs = [[-abs(val[0]), val[2], val[3]] for val in values]
    return chosen_pairs


def build_graph_and_clusters(chosen_pairs,df):
    """Build a graph from the pairs and perform clustering."""
    G = nx.Graph()
    for edge in chosen_pairs:
        G.add_edge(edge[1], edge[2], weight=edge[0])

    # Graph Laplacian
    L = np.diag([deg for node, deg in nx.degree(G)]) - nx.adjacency_matrix(G).A

    # Eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(L)
    vals, vecs = vals.real, vecs.real[:, np.argsort(vals.real)]

    range_n_clusters = range(5, 25)  # Change accordingly

    elbow, silhouette_avg = [], []
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters,random_state=0, n_init=25)
        cluster_labels = clusterer.fit_predict(vecs[:, 1:n_clusters])

        elbow.append(clusterer.inertia_)
        av_silhouette=silhouette_score(vecs[:, 1:n_clusters], cluster_labels)
        # print(av_silhouette)
        silhouette_avg.append(av_silhouette)
        # sample_silhouette_values = silhouette_samples(vecs[:, 1:n_clusters], cluster_labels)
        # for i in (cluster_labels):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            # ith_cluster_silhouette_values = \
            #     sample_silhouette_values[cluster_labels == i]

            # ith_cluster_silhouette_values.sort()
            # if ith_cluster_silhouette_values[-1]<av_silhouette:
            #     print(ith_cluster_silhouette_values)
            #     print(n_clusters,"False")
            #     break
        # print(sample_silhouette_values)

    optimal_clusters = np.where(np.diff(elbow) == max(np.diff(elbow)))[0][0] + 5
    optimal_k_silhouette = silhouette_avg.index(max(silhouette_avg)) + 5

    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f'The optimal number of clusters according to the silhouette method is {optimal_k_silhouette}')

    num_clusters = optimal_clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=20)
    kmeans.fit(vecs[:, 1:num_clusters])

    clusters = {i: [] for i in range(num_clusters)}
    for node_idx, cluster_id in enumerate(kmeans.labels_):
        clusters[cluster_id].append(list(G.nodes())[node_idx])

    for cluster_id, nodes in clusters.items():
        print(f"Cluster {cluster_id}: {nodes}\n {len(nodes)}")
        nr=max(2,len(nodes))
        fig,ax = plt.subplots(nrows=nr,figsize=(20,10))
        idx=[i for i in range(len(df))]
        for ix in range(len(nodes)):
            ax[ix].plot(idx,df[nodes[ix]].values)
            ax[ix].set_title(nodes[ix],fontsize=5)
        plt.show()
        # H=G.subgraph(nodes)
        # nx.draw(H, with_labels=True)
        # plt.show()
    return clusters


if __name__ == "__main__":
    df = pd.read_csv(argv[1])
    df1 = pd.read_csv(argv[1])
    df = standardize_dataframe(df)
    column_pairs = list(map(tuple, combinations(df.columns, 2)))
    values = [compute_cross_correlation(df, pair) for pair in column_pairs]
    values.sort(reverse=True)
    chosen_pairs = choose_pairs(values)
    clusters = build_graph_and_clusters(chosen_pairs,df1)


exit(0)
ctr=0
idx = [i for i in range(len(df))]

for i, cluster in enumerate(clusters):
    print(f'Cluster {i+1}: {cluster}')
    ctr+=len(cluster)
    n_plots=len(cluster)
    nrows=max(2,int(ceil(n_plots/3)))
    fig,ax=plt.subplots(nrows=nrows,ncols=3,figsize=(30,10))
    itr=iter(cluster)
    for j in range(n_plots):
        item=next(itr)
        ax[j//3,j%3].plot(idx,df[item].values)
        ax[j//3,j%3].set_title(item,fontsize=5)
        
    plt.xticks([])
    plt.show()

print("Total number of metrics clustered:",ctr, "against total",len(df.columns))
print(type(clusters[0]))
