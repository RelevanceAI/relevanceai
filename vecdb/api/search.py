from ..base import Base 

class Search(Base):
    def vector(self, dataset_id: str, multivector_query: list, positive_document_ids: dict={},
        negative_document_ids: dict={}, vector_operation="sum", approximation_depth=0,
        sum_fields=True, page_size=20, page=1, similarity_metric="cosine", facets=[], filters=[],
        min_score=0, select_fields=[], include_vector=False, include_count=True, asc=False, 
        keep_search_history=False, verbose: bool=True, output_format: str='json'):
        return self.make_http_request("services/search/vector", method="POST", parameters=
            {"dataset_id": dataset_id,
            "multivector_query": multivector_query,
            "positive_document_ids": positive_document_ids,
            "negative_document_ids": negative_document_ids,
            "vector_operation": vector_operation,
            "approximation_depth": approximation_depth,
            "sum_fields": sum_fields,
            "page_size": page_size,
            "page": page,
            "similarity_metric": similarity_metric,
            "facets": facets,
            "filters": filters,
            "min_score": min_score,
            "select_fields": select_fields,
            "include_vector": include_vector,
            "include_count": include_count,
            "asc": asc,
            "keep_search_history": keep_search_history
            }, output_format=output_format, verbose=verbose)
        
    def hybrid(self, dataset_id: str, multivector_query: list, 
        text: str, fields:list, 
        edit_distance: int=-1, ignore_spaces: bool=True,
        traditional_weight: float=0.075,
        page_size: int=20, page=1,
        similarity_metric="cosine", facets=[], filters=[],
        min_score=0, select_fields=[], include_vector=False, 
        include_count=True, asc=False, keep_search_history=False,
        verbose: bool=True, output_format: str='json'):
        return self.make_http_request("services/search/hybrid", method="POST",
            parameters={
                "dataset_id": dataset_id,
                "multivector_query": multivector_query,
                "text": text,
                "fields": fields,
                "page_size": page_size,
                "page": page,
                "similarity_metric": similarity_metric,
                "facets": facets,
                "filters": filters,
                "min_score": min_score,
                "select_fields": select_fields,
                "include_vector": include_vector,
                "include_count": include_count,
                "asc": asc,
                "keep_search_history": keep_search_history,
                "edit_distance": edit_distance,
                "ignore_spaces": ignore_spaces,
                "traditional_weight": traditional_weight
            }, output_format=output_format, verbose=verbose)

    def semantic(self, 
        dataset_id: str, 
        multivector_query: list, 
        fields: list, 
        text: str,
        page_size: int=20, page=1,
        similarity_metric="cosine", facets=[], filters=[],
        min_score=0, select_fields=[], include_vector=False, 
        include_count=True, asc=False, keep_search_history=False,
        verbose: bool=True, output_format: str='json'):
        return self.make_http_request("services/search/semantic", method="POST",
            parameters={
                "dataset_id": dataset_id,
                "multivector_query": multivector_query,
                "text": text,
                "fields": fields,
                "page_size": page_size,
                "page": page,
                "similarity_metric": similarity_metric,
                "facets": facets,
                "filters": filters,
                "min_score": min_score,
                "select_fields": select_fields,
                "include_vector": include_vector,
                "include_count": include_count,
                "asc": asc,
                "keep_search_history": keep_search_history
            }, output_format=output_format, verbose=verbose)

    def diversity(self, 
        dataset_id: str, 
        cluster_vector_field: str,
        multivector_query: list,
        positive_document_ids: dict={},
        negative_document_ids: dict={},
        vector_operation = "sum",
        approximation_depth: int=0,
        sum_fields: bool= True,
        page_size: int=20,
        page: int=1,
        similarity_metric = "cosine",
        facets=[], 
        filters=[],
        min_score=0, 
        select_fields=[], 
        include_vector=False, 
        include_count=True, 
        asc=False, 
        keep_search_history=False,
        hundred_scale: bool=False,
        search_history_id:str=None,
	    n_clusters: int=0,
	    n_init: int=5,
        n_iter: int=10,
        return_as_clusters: bool=False,
        verbose: bool=True, output_format: str='json'):
        return self.make_http_request("services/search/diversity", method="POST",
            parameters={
                "dataset_id": dataset_id,
                "multivector_query": multivector_query,
                "positive_document_ids": positive_document_ids,
                "negative_document_ids": negative_document_ids,
                "vector_operation": vector_operation,
                "approximation_depth": approximation_depth,
                "sum_fields": sum_fields,
                "page_size": page_size,
                "page": page,
                "similarity_metric": similarity_metric,
                "facets":facets,
                "filters":filters,
                "min_score": min_score,
                "select_fields": select_fields, 
                "include_vector": include_vector,
                "include_count": include_count, 
                "asc": asc, 
                "keep_search_history": keep_search_history,
                "hundred_scale": hundred_scale,
                "search_history_id": search_history_id,
                "cluster_vector_field": cluster_vector_field,
                "n_clusters": n_clusters,
                "n_init": n_init,
                "n_iter": n_iter,
                "return_as_clusters": return_as_clusters,
            }, output_format=output_format, verbose=verbose)
    
    def traditional(self, dataset_id: str, text: str,
        fields: list=[], edit_distance: int=-1,
        ignore_spaces: bool=True, page_size: int=29,
        page: int=1, select_fields: list=[],
        include_vector: bool=False, include_count: bool=True,
        asc: bool=False, keep_search_history: bool=False,
        search_history_id: str=None, verbose: bool=True,
        output_format: str='json'):
        return self.make_http_request('services/search/traditional', method="POST",
            parameters={
                "dataset_id": dataset_id,
                "text": text,
                "fields": fields,
                "edit_distance": edit_distance,
                "ignore_spaces": ignore_spaces,
                "page_size": page_size,
                "page": page,
                "select_fields": select_fields,
                "include_vector": include_vector,
                "include_count": include_count,
                "asc": asc,
                "keep_search_history": keep_search_history,
                "search_history_id": search_history_id
            }, output_format=output_format, verbose=verbose)

    def chunk(self, dataset_id, multivector_query, chunk_field, chunk_scoring = 'max', chunk_page_size:int = 3, chunk_page:int =1, approximation_depth:int = 0, sum_fields:bool = True, page_size:int = 20, page:int = 1, similarity_metric: str = 'cosine', 
        facets: list = [], filters:list = [], min_score: int = None, include_vector: bool  = False, include_count: bool  = True, asc: bool  = False, keep_search_history: bool = False, 
        verbose: bool=True, output_format: str='json'):
        return self.make_http_request('services/search/chunk', method="POST", 
                            parameters={"dataset_id": dataset_id,
                        "multivector_query": multivector_query,
                        "chunk_field": chunk_field,
                        "chunk_scoring": chunk_scoring,
                        "chunk_page_size": chunk_page_size,
                        "chunk_page": chunk_page,
                        "approximation_depth": approximation_depth,
                        "sum_fields": sum_fields,
                        "page_size": page_size,
                        "page": page,
                        "similarity_metric": similarity_metric,
                        "facets": facets,
                        "filters": filters,
                        "min_score": min_score,
                        "include_vector": include_vector,
                        "include_count": include_count,
                        "asc": asc,
                        "keep_search_history": keep_search_history
                            }, output_format=output_format, verbose=verbose)

    def multi_step_chunk(self, dataset_id, multivector_query, first_step_multivector_query, chunk_field, 
        chunk_scoring = 'max', chunk_page_size:int = 3, chunk_page:int =1, approximation_depth:int = 0, sum_fields:bool = True, page_size:int = 20, page:int = 1, similarity_metric: str = 'cosine', 
        facets: list = [], filters:list = [], min_score: int = None, include_vector: bool  = False, include_count: bool  = True, asc: bool  = False, keep_search_history: bool = False, first_step_page:int = 1, first_step_page_size: int = 20,
         verbose: bool=True, output_format: str='json'):
        return self.make_http_request('services/search/multistep_chunk', method="POST",
            parameters={"dataset_id": dataset_id,
                        "multivector_query": multivector_query,
                        "chunk_field": chunk_field,
                        "chunk_scoring": chunk_scoring,
                        "chunk_page_size": chunk_page_size,
                        "chunk_page": chunk_page,
                        "approximation_depth": approximation_depth,
                        "sum_fields": sum_fields,
                        "page_size": page_size,
                        "page": page,
                        "similarity_metric": similarity_metric,
                        "facets": facets,
                        "filters": filters,
                        "min_score": min_score,
                        "include_vector": include_vector,
                        "include_count": include_count,
                        "asc": asc,
                        "keep_search_history": keep_search_history,
                        "first_step_multivector_query": first_step_multivector_query,
                        "first_step_page": first_step_page,
                        "first_step_page_size": first_step_page_size
                            }, output_format=output_format, verbose=verbose)

    def advanced_chunk(self, dataset_ids, chunk_search_query, min_score: int = None, page_size: int = 20, include_vector: bool = False, select_fields: list = [], 
                        verbose: bool=True, output_format: str='json'):
        return self.make_http_request('services/search/advanced_chunk', method="POST",
            parameters={"dataset_ids": dataset_ids,
                        "chunk_search_query": chunk_search_query,
                        "page_size": page_size,
                        "min_score": min_score,
                        "include_vector": include_vector,
                        "select_fields": select_fields,
                            }, output_format=output_format, verbose=verbose)

def advanced_multistep_chunk(
    self,
    dataset_ids: list,
    first_step_query: list,
    first_step_text: str,
    first_step_fields: list,
    chunk_search_query: list,
    first_step_edit_distance: int = -1,
    first_step_ignore_space: bool = True,
    first_step_traditional_weight: float = 0.075,
    first_step_approximation_depth: int = 0,
    first_step_sum_fields: bool = True,
    first_step_filters: list = [],
    first_step_page_size: int = 50,
    include_count: bool = True,
    min_score: int= 0,
    page_size: int= 20,
    include_vector: bool = False,
    select_fields: list = [],
    verbose: bool=True,
    output_format: str='json'):
    return self.make_http_request('services/search/advanced_multistep_chunk', method="POST",
            parameters={"dataset_ids": dataset_ids,
                        "first_step_query": first_step_query,
                        "first_step_text": first_step_text,
                        "first_step_fields": first_step_fields,
                        "chunk_search_query": chunk_search_query,
                        "first_step_edit_distance": first_step_edit_distance,
                        "first_step_ignore_space": first_step_ignore_space,
                        "first_step_traditional_weight": first_step_traditional_weight,
                        "first_step_approximation_depth": first_step_approximation_depth,
                        "first_step_sum_fields": first_step_sum_fields,
                        "first_step_filters": first_step_filters,
                        "first_step_page_size": first_step_page_size,
                        "include_count": include_count,
                        "min_score": min_score,
                        "page_size": page_size,
                        "include_vector": include_vector,
                        "select_fields": select_fields}, output_format=output_format, verbose=verbose)
