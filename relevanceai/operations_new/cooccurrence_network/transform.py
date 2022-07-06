"""
Build a concept graph based on the documents in your dataset
"""
import sys

import numpy as np

from sklearn.cluster import AgglomerativeClustering

from relevanceai.operations_new.transform_base import TransformBase
from relevanceai.operations_new.processing.text.clean.transform import CleanTextTransform
from relevanceai.constants.stopwords import STOPWORDS


def preprocess(data):
    stopwords = STOPWORDS
    cleaner = CleanTextTransform(text_fields=[], output_fields=[], lower=True,
                                 lemmatize=True, remove_stopwords=stopwords)
    res = []
    for doc in data:
        res.append(cleaner.clean_text(doc).split(' '))

    return res


class WordDictionary():
    def __init__(self, docs):
        self.id2word = []
        self.word2id = dict()
        self.id2dfs = dict()
        for doc in docs:
            words = set()
            for word in doc:
                if word not in self.word2id:
                    self.id2word.append(word)
                    self.word2id[word] = len(self.id2word)
                if word not in words:
                    words.add(word)
                    self.id2dfs[self.word2id[word]] = self.id2dfs.get(self.word2id[word], 0) + 1


class ConceptGraphTransform(TransformBase):
    def __init__(
            self,
            max_number_of_clusters=15,
            min_number_of_clusters=3,
            number_of_concepts=100,
            **kwargs,
    ):
        """
        Sentiment Ops.

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

    def findMaxVertex(self, visited, weights):
        # Stores the index of max-weight vertex
        # from set of unvisited vertices
        index = -1;

        # Stores the maximum weight from
        # the set of unvisited vertices
        maxW = -sys.maxsize;

        # Iterate over all possible
        # Nodes of a graph
        for i in range(self.number_of_concepts):

            # If the current Node is unvisited
            # and weight of current vertex is
            # greater than maxW
            if (visited[i] == False and weights[i] > maxW):
                # Update maxW
                maxW = weights[i];

                # Update index
                index = i;
        return index;

    # Function to find the maximum spanning tree
    def maximumSpanningTree(self, graph):

        # visited[i]:Check if vertex i
        # is visited or not
        visited = [True] * self.number_of_concepts;

        # weights[i]: Stores maximum weight of
        # graph to connect an edge with i
        weights = [0] * self.number_of_concepts;

        # parent[i]: Stores the parent Node
        # of vertex i
        parent = [0] * self.number_of_concepts;

        # Initialize weights as -INFINITE,
        # and visited of a Node as False
        for i in range(self.number_of_concepts):
            visited[i] = False;
            weights[i] = -sys.maxsize;

        # Include 1st vertex in
        # maximum spanning tree
        weights[0] = sys.maxsize;
        parent[0] = -1;

        # Search for other (V-1) vertices
        # and build a tree
        for i in range(self.number_of_concepts - 1):

            # Stores index of max-weight vertex
            # from a set of unvisited vertex
            maxVertexIndex = self.findMaxVertex(visited, weights);

            # Mark that vertex as visited
            visited[maxVertexIndex] = True;

            # Update adjacent vertices of
            # the current visited vertex
            for j in range(self.number_of_concepts):

                # If there is an edge between j
                # and current visited vertex and
                # also j is unvisited vertex
                if (graph[j][maxVertexIndex] != 0 and visited[j] == False):

                    # If graph[v][x] is
                    # greater than weight[v]
                    if (graph[j][maxVertexIndex] > weights[j]):
                        # Update weights[j]
                        weights[j] = graph[j][maxVertexIndex];

                        # Update parent[j]
                        parent[j] = maxVertexIndex;

        return parent

    def transform(self, documents):
        # For each document, update the field
        docs = [d['content'] for d in documents]

        cleaned_texts = preprocess(docs)

        # Create Dictionary
        word_dict = WordDictionary(cleaned_texts)
        # Create Corpus
        texts = cleaned_texts

        top_ids = sorted(word_dict.id2dfs, key=word_dict.id2dfs.get, reverse=True)[:self.number_of_concepts]

        word_count_mat = []
        for i, text in enumerate(texts):
            row = [0] * self.number_of_concepts
            for word in text:
                id = word_dict.word2id[word]
                if id in top_ids:
                    row[top_ids.index(id)] = 1
            word_count_mat.append(row)
        word_count_mat = np.array(word_count_mat)

        concurrence_matrix = np.dot(word_count_mat.transpose(), word_count_mat)

        mst = self.maximumSpanningTree(concurrence_matrix)

        mat = concurrence_matrix[0][0] - concurrence_matrix

        vertexes = []
        for i, id in enumerate(top_ids):
            v = {'word': word_dict.id2word[id], 'rank': i, 'count': word_dict.id2dfs.get(id)}
            vertexes.append(v)

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
            uniqueValues = np.unique(labels[:, i])
            label2label = dict(zip(uniqueValues, range(len(uniqueValues))))
            for j, v in enumerate(labels[:, i]):
                labels[j, i] = label2label[v]

        for i in range(self.min_number_of_clusters, self.max_number_of_clusters + 1):
            for j in range(self.number_of_concepts):
                vertexes[j]['label_{}'.format(i)] = labels[j, self.number_of_concepts - i]

        edges = []
        for sour, dest in enumerate(mst[1:]):
            edges.append((word_dict.id2word[top_ids[dest]], word_dict.id2word[top_ids[sour]]))

        return vertexes, edges

    @property
    def name(self):
        return "concept_graph"
