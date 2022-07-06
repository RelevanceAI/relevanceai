from relevanceai.operations_new.concept_graph.transform import ConceptGraphTransform
import pickle
from sklearn.datasets import fetch_20newsgroups
import graphviz


# def draw_graph(graph, n_clusters):
#     vertexes, edges = graph
#     dot = graphviz.Graph('test',engine='neato')
#
#     for id in top_ids:
#       dot.node(id2word[id])
#
#     for sour, dest in enumerate(mst[1:]):
#         dot.edge(id2word[top_ids[dest]], id2word[top_ids[sour]])
#
#     for i, label in enumerate(clustering.labels_):
#         dot.node(id2word[top_ids[i]], color=colors[label])

if __name__ == '__main__':
    X, y = fetch_20newsgroups(subset = 'test', shuffle = False, remove=('headers', 'footers', 'quotes'), return_X_y = True)
    data = []
    for n, sent in enumerate(X[:1000]):
        doc = " ".join(sent.replace('\n', ' ').split())
        data.append({'content':doc})

    # print(data[9]['content'])



    model = ConceptGraphTransform(max_number_of_clusters = 15, min_number_of_clusters = 3, number_of_concepts = 100)
    graph = model.transform(data)

    print(graph)