"""Report for grading
"""


def get_silhouette_grade(silhouette_score):
    if silhouette_score > 0.9:
        return "S"
    if silhouette_score > 0.7:
        return "A"
    if silhouette_score > 0.6:
        return "B"
    if silhouette_score > 0.4:
        return "C"
    if silhouette_score > 0.2:
        return "D"
    if silhouette_score > 0.0:
        return "D"
    if silhouette_score > -0.2:
        return "E"
    if silhouette_score > -0.4:
        return "E"
    if silhouette_score > -0.6:
        return "F"
    if silhouette_score > -0.8:
        return "F"
    return "F"


def dunn_index(X, y):
    raise NotImplementedError("Dunn index not supported at the moment.")


# def get_dunn_index(score):
#     _internal_report = {}
#     _internal_report["overall"]["dunn_index"] = self.dunn_index(
#         min_centroid_distance, max_centroid_distance
#     )
#     pass


def get_dunn_index_grade(score):
    if score > 0.9:
        return "AA"
    if score > 0.8:
        return "A"
    if score > 0.7:
        return "B"
    if score > 0.6:
        return "C"
    if score > 0.5:
        return "D"
    if score > 0.4:
        return "E"
    return "F"
