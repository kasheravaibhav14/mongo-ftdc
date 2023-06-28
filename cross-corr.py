import pandas as pd
import numpy as np
from itertools import combinations
import networkx as nx
from sys import argv
import warnings
from math import ceil
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans

warnings.simplefilter("ignore")

def standardize_dataframe(df):
    return (df - df.mean()) / df.std()

def compute_cross_correlation(df, pair):
    xcorrelation = np.correlate(df[pair[0]], df[pair[1]], 'full')
    autocorr_1 = np.correlate(df[pair[0]], df[pair[0]], 'full')
    autocorr_2 = np.correlate(df[pair[1]], df[pair[1]], 'full')
    xcorrelation /= np.sqrt(autocorr_1[len(df[pair[0]])-1] * autocorr_2[len(df[pair[1]])-1]) # if both timeseries are of same size, it calculates to number of timeseries

    lags = np.arange(-len(df[pair[0]]) + 1, len(df[pair[1]]))
    maxlag_param = 150
    indices = np.where((lags >= -maxlag_param) & (lags <= maxlag_param))

    restricted_lags = lags[indices]
    restricted_xcorrelation = xcorrelation[indices]

    maxlag = restricted_lags[np.argmax(restricted_xcorrelation)]
    maxval = np.max(restricted_xcorrelation)
    if maxval<0:
        maxval+=abs(maxlag)
    return (maxval,-abs(maxlag),pair[0],pair[1])

def choose_pairs(values):
    chosen_pairs = []
    mets = set()
    
    for val in values:
        # if val[2] in mets and val[3] in mets:
        #     continue
        chosen_pairs.append([-abs(val[0]), val[2], val[3]])
        mets.add(val[2])
        mets.add(val[3])
    
    return chosen_pairs

def build_graph_and_clusters(chosen_pairs):
    G = nx.Graph()
    for edge in chosen_pairs:
        G.add_edge(edge[1],edge[2],weight=edge[0])
    # G.add_edges_from(chosen_pairs)
    A = nx.adjacency_matrix(G).A  # Convert adjacency matrix to numpy array

# Print Adjacency matrix
    print("Adjacency Matrix:")
    print(A)
    D = np.diag([deg for node, deg in nx.degree(G)])

# Graph Laplacian
    L = D - A

    # Eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(L)
    vals=vals.real
    vecs=vecs.real

    # Sort these based on the eigenvalues
    vecs = vecs[:, np.argsort(vals)]
    vals = vals[np.argsort(vals)]

    # kmeans on first three vectors with nonzero eigenvalues
    range_n_clusters = range(5, 15)  # Change accordingly

# Variables to store results
    elbow = []
    delta_elbow = []  # Difference in WCSS

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(vecs[:, 1:4])

        # The total sum of squares
        wcss = clusterer.inertia_
        elbow.append(wcss)

    # Calculate the difference in WCSS
    print(elbow)
    second_derivative = np.diff(elbow, 2)
    # Find the optimal number of clusters
    # The "elbow" is identified as the point where the WCSS decrease rate changes significantly
    optimal_clusters = np.argmax(second_derivative) + 3

    print(f"Optimal number of clusters: {optimal_clusters}")
    num_clusters=optimal_clusters
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(vecs[:, 1:num_clusters])
    colors = kmeans.labels_

    clusters = {i: [] for i in range(num_clusters)}  # Initialize empty lists for each cluster

    # Fill the lists based on the assigned cluster ID for each node
    for node_idx, cluster_id in enumerate(colors):
        clusters[cluster_id].append(list(G.nodes())[node_idx])

    # Print the clusters
    for cluster_id, nodes in clusters.items():
        print(f"Cluster {cluster_id}: {nodes}")
   
    return clusters

df = pd.read_csv(argv[1])
df1 = pd.read_csv(argv[1])
df = standardize_dataframe(df)

column_pairs = list(map(tuple, combinations(df.columns, 2)))

values = [compute_cross_correlation(df, pair) for pair in column_pairs]
values.sort(reverse=True)

chosen_pairs = choose_pairs(values)

clusters = build_graph_and_clusters(chosen_pairs)
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
