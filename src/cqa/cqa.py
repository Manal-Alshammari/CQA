from math import sqrt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import rand_score
import yake


def _generate_keywords(file):
    keywords_extractor = yake.KeywordExtractor()
    keywords_scored = keywords_extractor.extract_keywords(file)
    # This a list for all keywords WITHOUT its scores- using YAKE algorithm
    yake_keywords = [word[0] for word in keywords_scored]
    return yake_keywords


def _get_max_similarity(tallest_list, shortest_list):
    model_name = 'sentence-transformers/bert-large-nli-mean-tokens'
    model = SentenceTransformer(model_name)
    windows_num = (len(tallest_list) - len(shortest_list)) + 1
    number_of_clusters = int(sqrt((len(tallest_list) + len(shortest_list)) / 2))

    rand_index_list = []

    for i in range(windows_num):
        window = []
        for j in range(len(shortest_list)):
            window.append(tallest_list[(j + i)])
        window_embedding = model.encode(window)
        s_embedding = model.encode(shortest_list)
        k_means = KMeans(n_clusters=number_of_clusters, random_state=0).fit(window_embedding)
        prediction = k_means.predict(s_embedding)
        rand_index_list.append(rand_score(k_means.labels_, prediction))
    max_similarity = max(rand_index_list)
    return max_similarity


def get_alignment(clo_file_path, so_file_path):
    # Open the two files
    clo_file = open(clo_file_path, encoding="utf8").read()
    so_file = open(so_file_path, encoding="utf8").read()
    # Extract the KeyWords
    clo_keywords = _generate_keywords(clo_file)
    so_keywords = _generate_keywords(so_file)

    if len(clo_keywords) > len(so_keywords):
        max_similarity = _get_max_similarity(clo_keywords, so_keywords)
    else:
        max_similarity = _get_max_similarity(so_keywords, clo_keywords)

    return round(max_similarity * 100)
