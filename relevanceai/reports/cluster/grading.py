"""Report for grading
"""


def get_silhouette_grade(silhouette_score):
    # grades = ["F"] + [f"{grade}{sign}" for grade in ["E", "D", "C", "B", "A"] for sign in ["-", "", "+"]] + ["S"]
    grades = ["F", "E", "D", "C", "B", "A", "S"]
    scores = [(2 * i) / len(grades) for i in range(1, len(grades) + 1)]
    for score, grade in zip(scores, grades):
        if (silhouette_score + 1) < score:
            return grade
    return "N/A"


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
