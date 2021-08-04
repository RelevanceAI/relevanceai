from ..base import Base 

class Search(Base):
    def search(self, dataset_id: str, multivector_query: list, positive_document_ids: dict={},
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
        query: str, fields:list, page_size: int=20, page=1,
        similarity_metric="cosine", facets=[], filters=[],
        min_score=0, select_fields=[], include_vector=False, 
        include_count=True, asc=False, keep_search_history=False,
        verbose: bool=True, output_format: str='json'):
        return self.make_http_request("services/search/hybrid", method="POST",
            parameters={
                "dataset_id": dataset_id,
                "multivector_query": multivector_query,
                "text": query,
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
