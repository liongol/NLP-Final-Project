from compare_clustering_solutions import evaluate_clustering

import json
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

import time
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
PLOT = False

### Helper functions ###

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    '''
    param: v1: np.ndarray, a vector
    param: v2: np.ndarray, a vector
    return: float, the cosine similarity between v1 and v2
    '''
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def ngrams(input, n):
    '''
    param: input: str, a string
    param: n: int, the number of words in the n-gram
    return: list[str], a list of n-grams
    '''
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


### End of helper functions ###

def read_data(file_name: str) -> list[str]:
    '''
    param: file_name: str, the name of the csv file to read
    return: list[str], a list of strings, each string is a request from the original csv file
    '''

    df = pd.read_csv(file_name)
    return df['request'].tolist()

def get_embeddings(requests: list[str], model_name: str = 'all-MiniLM-L6-v2') -> list[np.ndarray]:
    '''
    param: requests: list[str], a list of requests
    return: list[np.ndarray], a list of embeddings, each embedding is a numpy array
    '''
    model = SentenceTransformer(model_name)
    embeddings = model.encode(requests)

    if PLOT:
        pca = PCA(n_components=2)
        pca.fit(embeddings)
        embeddings_2d = pca.transform(embeddings)
        plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1])
        plt.title('Embeddings')
        plt.show()

    return embeddings

def cluster_with_kmeans_variation(embeddings: list[np.ndarray], similarity_threshold: float = 0.5, min_cluster_size: int = 10) -> list[dict[list[int], np.ndarray]]:
    '''
    param: embeddings: list[np.ndarray], a list of embeddings
    param: similarity_threshold: float, the maximum distance between two samples for one to be considered as in the neighborhood of the other
    param: min_cluster_size: int, the minimum number of requests in a cluster
    return: list[list[int]], a list of lists, each list is a cluster, each element in the list is the request_id of the request (we know that the lists maintain the order of the requests)
    '''
    clusters = []

    clusters = []

    for i, embedding in enumerate(embeddings):  # iterate over all requests
        assigned = False
        for cluster in clusters:
            if cosine_similarity(embedding, cluster['centroid']) >= similarity_threshold:  # if the request is similar to the centroid of an existing cluster:
                cluster['request_ids'].append(i)  # add the request to the cluster
                cluster['centroid'] = np.mean([embeddings[request_id] for request_id in cluster['request_ids']],
                                              axis=0)  # update the centroid of the cluster
                assigned = True
                break
        if not assigned:  # if the request was not assigned to any cluster, create a new cluster
            clusters.append({'request_ids': [i], 'centroid': embedding})

    result = [cluster for cluster in clusters if len(cluster['request_ids']) >= int(min_cluster_size)] # return only clusters with at least min_cluster_size requests

    if PLOT:
        pca = PCA(n_components=2)
        pca.fit(embeddings)
        embeddings_2d = pca.transform(embeddings)
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'grey']
        for i, cluster in enumerate(result):
            plt.scatter(embeddings_2d[cluster['request_ids'],0], embeddings_2d[cluster['request_ids'],1], c=colors[i%len(colors)])
        plt.title('Clusters')
        plt.show()

    return result

def get_k_representatives(orig_clusters: list[dict[list[int], np.ndarray]], embedded_requests: list[np.ndarray], k: int) -> list[list[int]]:
    '''
    param: orig_clusters: list[dict[list[int], np.ndarray]], a list of clusters
    param: k: int, the number of clusters to return
    return: list[list[int]], a list of lists, each list is a cluster, each element in the list is the request_id of the request (there will be k requst_ids in each list)
    For that, here is the general algorithm:
    1. for each cluster:
    1.1. compute k-means on the cluster (with k=k)
    2. within each cluster, take the requst that is the furthest (in average) from all the other cluser centroids
    3. return the list of request_ids
    '''
    clusters = []
    for cluster in orig_clusters:
        clustered_ids = cluster['request_ids']
        if len(clustered_ids) <= int(k):
            clusters.append(clustered_ids)
        else:
            kmeans = KMeans(n_clusters=int(k), n_init='auto').fit(
                [embedded_requests[request_id] for request_id in clustered_ids])
            centroids = kmeans.cluster_centers_
            distances = []
            for i, request_id in enumerate(clustered_ids):
                distances.append(
                    (request_id, np.linalg.norm(embedded_requests[request_id] - centroids[kmeans.labels_[i]])))
            distances.sort(key=lambda x: x[1], reverse=True)
            clusters.append([request_id for request_id, _ in distances[:int(k)]])

        if PLOT:
            # for each cluster, plot the clusters embeddings int blue and color the representatives in red in 2D space using PCA
            #plot only the first 5 clusters
            if len(clusters) > 5:
                break
            pca = PCA(n_components=2)
            pca.fit([embedded_requests[request_id] for request_id in clustered_ids])
            embeddings_2d = pca.transform([embedded_requests[request_id] for request_id in clustered_ids])
            plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c='b')
            representatives_2d = pca.transform([embedded_requests[request_id] for request_id in clusters[-1]])
            plt.scatter(representatives_2d[:,0], representatives_2d[:,1], c='r')
            plt.title('Cluster')
            plt.show()

    return clusters


def get_clusters_names(extracted_requests: list[str], clusters: list[dict[list[int], np.ndarray]], model_name: str = 'all-MiniLM-L6-v2') ->list[str]:
    '''
    param: extracted_requests: list[str], a list of requests
    param: clusters: list[dict[list[int], np.ndarray]], a list of clusters
    param: model_name: str, the name of the model to use for the embedding
    return: list[str], a list of strings, each string is the name of the cluster
    '''
    cluster_names = []

    for cluster in clusters:
        list_of_actual_requests = [extracted_requests[request_id] for request_id in
                                   cluster['request_ids']]  # get the actual requests from the request_ids

        # create n-grams of the requests (2,3,4,5)
        n_grams = []
        for request in list_of_actual_requests:
            for n in range(2, 6):
                n_grams.append(ngrams(request, n))

        # remove empty lists
        n_grams = [n_gram for n_gram in n_grams if n_gram != []]

        all_actual_n_grams = []
        # iterate over the list of lists:
        for clustered_ngrams in n_grams:
            for ngram in clustered_ngrams:
                all_actual_n_grams.append(' '.join(ngram))  # join the words in the n-gram to form all the n-grams

        # embed the n-grams
        model = SentenceTransformer(model_name)  # load the model
        embedded_n_grams = model.encode(all_actual_n_grams)  # embed the n-grams

        # check the cosine similarity between the n-grams and the centroid of the cluster
        centroid = cluster['centroid']
        similarities = []
        for n_gram in embedded_n_grams:
            similarities.append(cosine_similarity(centroid,
                                                  n_gram))  # again, we use cosine similarity to measure the similarity between the n-gram and the centroid (we assume that the centroid is the most representative request in the cluster and the semantic meaning of the cluster is the semantic meaning of the centroid)

        # return the n-gram with the highest similarity
        cluster_name = all_actual_n_grams[similarities.index(max(similarities))]

        # if the cluster name ends with a question mark, we remove the question mark and add "Questions about" at the beginning
        if cluster_name[-1] == '?':
            cluster_name = cluster_name[:-1]
            cluster_name = 'Questions about ' + cluster_name

        cluster_names.append(cluster_name)  # add the cluster name to the list of cluster names

    return cluster_names

def dump_to_json(clustered_ids: list[list[int]], extracted_requests: list[str], cluster_names: list[str], k_representatives: list[list[str]], json_file_to_store_to: str) -> None:
    '''
    param: clustered_ids: list[list[int]], a list of lists, each list is a cluster, each element in the list is the request_id of the request (we know that i gives us embeddings[i]
    param: extracted_requests: list[str], a list of requests, each request is a string
    param: cluster_names: list[str], a list of cluster names (in the same order as the clusters)
    param: k_representatives: list[list[int]], a list of lists, each list is a cluster, each element in the list is a request_id of a representative request
    param: json_file_to_store_to: str, the path to the json file to store the data to
    at the end, we have a json file in the same format as the json file we get to test our model on
    '''
    cluster_list = []
    for i, cluster in enumerate(clustered_ids):
        cluster_dict = {}
        cluster_dict['cluster_name'] = cluster_names[i],
        cluster_dict['representative_sentences'] = [extracted_requests[request_id] for request_id in k_representatives[i]]
        cluster_dict['requests'] = [extracted_requests[request_id] for request_id in cluster]
        cluster_list.append(cluster_dict)
    unclustered = [extracted_requests[i] for i in range(len(extracted_requests)) if
                   i not in [request_id for cluster in clustered_ids for request_id in cluster]]
    data = {}
    data['cluster_list'] = cluster_list
    data['unclustered'] = unclustered
    with open(json_file_to_store_to, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def analyze_unrecognized_requests(data_file, output_file, num_rep, min_size):
    '''
    data_file: str, the name of the csv file to read (has all the requests)
    output_file: str, the name of the json file to write the results to
    num_rep: int, the number of representatives we want to find for each cluster
    min_size: int, the minimum size of a cluster to be considered a valid cluster
    '''
    start_time = time.time()

    print("Input file: ", data_file)
    print("Starting...")

    extracted_requests = read_data(data_file)

    # hyperparameters, we chose them after performing cross-validation
    model_name = 'all-MiniLM-L6-v2'
    similarity_threshold = 0.64

    embedded_requests = get_embeddings(extracted_requests, model_name)
    embedding_time = int((time.time() - start_time)/60)
    print("Embedding done within: %s minutes and %s seconds" % (embedding_time, int((time.time() - start_time)%60)))

    clusters = cluster_with_kmeans_variation(embedded_requests, similarity_threshold, min_size)
    clustering_time = int((time.time() - start_time)/60) - embedding_time
    print("Clustering done within: %s minutes and %s seconds" % (clustering_time, int((time.time() - start_time)%60)))

    clustered_ids = [cluster['request_ids'] for cluster in clusters]

    k_representatives = get_k_representatives(clusters, embedded_requests, num_rep)
    get_representatives_time = int((time.time() - start_time)/60) - embedding_time - clustering_time
    print("Getting representatives done within: %s minutes and %s seconds" % (get_representatives_time, int((time.time() - start_time)%60)))

    clusters_names = get_clusters_names(extracted_requests, clusters, model_name)
    get_names_time = int((time.time() - start_time)/60) - embedding_time - clustering_time - get_representatives_time
    print("Getting names done within: %s minutes and %s seconds" % (get_names_time, int((time.time() - start_time)%60)))

    dump_to_json(clustered_ids, extracted_requests, clusters_names, k_representatives, output_file)

    print("Overall time: %s minutes and %s seconds" % (int((time.time() - start_time)/60), int((time.time() - start_time)%60)))

if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                    config['output_file'],
                                    config['num_of_representatives'],
                                    config['min_cluster_size'])


    evaluate_clustering(config['output_file'], config['example_solution_file'])
