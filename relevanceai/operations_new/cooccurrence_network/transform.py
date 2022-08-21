"""
Build a co-occurrence network based on the documents in your dataset
"""
import sys

from collections import defaultdict
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from relevanceai.operations_new.transform_base import TransformBase
from relevanceai.operations_new.processing.text.clean.transform import CleanTextTransform
from relevanceai.constants.stopwords import STOPWORDS


def preprocess(data, stopwords_list=[]):
    # stopwords_list = STOPWORDS # stopwords is a list of strings.
    # stopwords_list += stopwords
    cleaner = CleanTextTransform(text_fields=[], output_fields=[], lower=True,
                                 lemmatize=True, remove_stopwords=STOPWORDS + stopwords_list)
    res = []
    for doc in data:
        res.append(cleaner.clean_text(doc).split(' '))
    return res


class WordDictionary():
    def __init__(self, docs):
        self.id2word = []
        self.word2id = dict()
        self.id2dfs = dict()
        self.id2docs = defaultdict(list)
        self.doc2ids = defaultdict(list)
        for doc_id, doc in enumerate(docs):
            words = set(doc)
            for word in words:
                if word not in self.word2id:
                    self.word2id[word] = len(self.id2word)
                    self.id2word.append(word)
                    self.id2dfs[self.word2id[word]] = 1
                else:
                    self.id2dfs[self.word2id[word]] = self.id2dfs.get(self.word2id[word]) + 1
                self.doc2ids[doc_id].append(self.word2id[word])
                self.id2docs[self.word2id[word]].append(doc_id)

    def get_df_table(self):
        return self.id2dfs

    def get_ids(self, doc_id):
        return self.doc2ids[doc_id]

    def get_docs(self, word: str):
        return self.id2docs[self.word2id[word]]

    def get_word(self, id):
        return self.id2word[id]

    def get_id(self, word: str):
        return self.word2id[word]

    # return updated df_table based on only the documents contains word
    def update_df_table(self, word: str):
        df_table = dict()
        for doc_id in self.get_docs(word):
            for word_id in self.get_ids(doc_id):
                if word_id not in df_table:
                    df_table[word_id] = 1
                else:
                    df_table[word_id] = df_table[word_id] + 1
        return df_table


class CoOccurNetTransform(TransformBase):
    def __init__(
            self,
            max_number_of_clusters=15,
            min_number_of_clusters=3,
            number_of_concepts=100,
            **kwargs,
    ):
        """
        Parameters
        -------------

        max_number_of_clusters: int

        min_number_of_clusters: int

        number_of_concepts: int
        """

        self.max_number_of_clusters = max_number_of_clusters
        self.min_number_of_clusters = min_number_of_clusters
        self.number_of_concepts = number_of_concepts
        for k, v in kwargs.items():
            setattr(self, k, v)

    def find_max_vertex(self, visited, weights):
        # Stores the index of max-weight vertex
        # from set of unvisited vertices
        index = -1

        # Stores the maximum weight from
        # the set of unvisited vertices
        max_weight = -sys.maxsize

        # Iterate over all possible
        # Nodes of a graph
        for i in range(self.number_of_concepts):
            # If the current Node is unvisited
            # and weight of current vertex is
            # greater than max_weight
            if visited[i] is False and weights[i] > max_weight:
                # Update max_weight
                max_weight = weights[i]
                # Update index
                index = i
        return index

    # Function to find the maximum spanning tree
    def maximum_spanning_tree(self, graph):
        # visited[i]:Check if vertex i
        # is visited or not
        visited = [True] * self.number_of_concepts

        # weights[i]: Stores maximum weight of
        # graph to connect an edge with i
        weights = [0] * self.number_of_concepts

        # parent[i]: Stores the parent Node
        # of vertex i
        parent = [0] * self.number_of_concepts

        # Initialize weights as -INFINITE,
        # and visited of a Node as False
        for i in range(self.number_of_concepts):
            visited[i] = False
            weights[i] = -sys.maxsize

        # Include 1st vertex in
        # maximum spanning tree
        weights[0] = sys.maxsize
        parent[0] = -1

        # Search for other (V-1) vertices
        # and build a tree
        for i in range(self.number_of_concepts - 1):

            # Stores index of max-weight vertex
            # from a set of unvisited vertex
            max_vertex_index = self.find_max_vertex(visited, weights)

            # Mark that vertex as visited
            visited[max_vertex_index] = True

            # Update adjacent vertices of
            # the current visited vertex
            for j in range(self.number_of_concepts):

                # If there is an edge between j
                # and current visited vertex and
                # also j is unvisited vertex
                if graph[j][max_vertex_index] != 0 and visited[j] == False:

                    # If graph[v][x] is
                    # greater than weight[v]
                    if graph[j][max_vertex_index] > weights[j]:
                        # Update weights[j]
                        weights[j] = graph[j][max_vertex_index]

                        # Update parent[j]
                        parent[j] = max_vertex_index
        return parent

    def concurrence_matrix(self, top_ids, texts, word_dict):
        word_count_mat = []
        for i, text in enumerate(texts):
            row = [0] * self.number_of_concepts
            for word in text:
                id = word_dict.word2id[word]
                if id in top_ids:
                    row[top_ids.index(id)] = 1
            word_count_mat.append(row)
        word_count_mat = np.array(word_count_mat)

        return np.dot(word_count_mat.transpose(), word_count_mat)

    def get_clusters_labels(self, mat):
        labels = np.zeros((self.number_of_concepts, self.number_of_concepts), dtype=int)

        for i in range(self.number_of_concepts):
            labels[i, 0] = i
        for i, merged in enumerate(
                AgglomerativeClustering(affinity='precomputed', n_clusters=i, linkage='complete').fit(mat).children_):
            new_id = i + self.number_of_concepts
            for j in range(self.number_of_concepts):
                if labels[j, i] in merged:
                    labels[j, i + 1] = new_id
                else:
                    labels[j, i + 1] = labels[j, i]

        for i in range(self.number_of_concepts):
            unique_values = np.unique(labels[:, i])
            label2label = dict(zip(unique_values, range(len(unique_values))))
            for j, v in enumerate(labels[:, i]):
                labels[j, i] = label2label[v]

        return labels

    def transform(self, documents, text_field='content', stopwords_list=[], center_word=None):
        # For each document, update the field
        docs = [d[text_field] for d in documents]
        cleaned_texts = preprocess(docs, stopwords_list)

        # Create Dictionary
        word_dict = WordDictionary(cleaned_texts)

        # Create Corpus
        if center_word is None:
            texts = cleaned_texts
            df_table = word_dict.get_df_table()
        else:
            doc_id_list = word_dict.get_docs(center_word)
            texts = [cleaned_texts[i] for i in doc_id_list]
            df_table = word_dict.update_df_table(center_word)

        top_ids = sorted(df_table, key=df_table.get, reverse=True)[:self.number_of_concepts]
        co_occur_mat = self.concurrence_matrix(top_ids, texts, word_dict)
        mst = self.maximum_spanning_tree(co_occur_mat)

        vertexes = []
        for i, id in enumerate(top_ids):
            v = {'word': word_dict.id2word[id], 'rank': i, 'count': df_table.get(id)}
            vertexes.append(v)

        labels = self.get_clusters_labels(co_occur_mat[0][0] - co_occur_mat)
        for i in range(self.min_number_of_clusters, self.max_number_of_clusters + 1):
            for j in range(self.number_of_concepts):
                vertexes[j]['label_{}'.format(i)] = labels[j, self.number_of_concepts - i]

        edges = []
        for sour, dest in enumerate(mst[1:]):
            edges.append((word_dict.id2word[top_ids[dest]], word_dict.id2word[top_ids[sour+1]]))

        return vertexes, edges

    @property
    def name(self):
        return "co_occurrence_network"
