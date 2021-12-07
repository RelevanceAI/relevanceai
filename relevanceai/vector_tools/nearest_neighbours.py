import scipy.spatial.distance as spatial_distance
from doc_utils.doc_utils import DocUtils
from relevanceai.vector_tools.constants import NEAREST_NEIGHBOURS

def get_nearest_neighbours(
    docs: list, 
    vector: list,
    vector_field: str,
    distance_measure_mode: NEAREST_NEIGHBOURS= 'cosine', 
    callable_distance = None
    ): 

    if callable_distance:
        return sorted(docs, key=lambda x: [callable_distance(i, vector) for i in DocUtils.get_field_across_documents(vector_field, docs)])

    elif distance_measure_mode == 'cosine':
        return sorted(docs, key=lambda x: [spatial_distance.cosine(i, vector) for i in DocUtils.get_field_across_documents(vector_field, docs)])

    elif distance_measure_mode == 'l2':
        return sorted(docs, key=lambda x: [spatial_distance.euclidean(i, vector) for i in DocUtils.get_field_across_documents(vector_field, docs)])

    else:
        raise ValueError('Need valid distance measure mode or callable distance')