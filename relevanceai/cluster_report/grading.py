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
