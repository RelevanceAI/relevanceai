class ExperimentHelper():
    def __init__(self, last_search, categories = ['chunk', 'vector', 'diversity', 'traditional']):
        self.last_search = last_search
        self.categories = categories

        self.traditional_search_doc = "https://docs.relevance.ai/docs/better-text-search-with-hybrid"
        self.vector_search_doc = "https://docs.relevance.ai/docs/pure-word-matching-pure-vector-search-or-combination-of-both"
        self.diversity_search_doc = "https://docs.relevance.ai/docs/better-text-search-diversified-search-results"
        self.hybrid_search_doc = "https://docs.relevance.ai/docs/pure-word-matching-pure-vector-search-or-combination-of-both-1"
        self.semantic_search_doc = "https://docs.relevance.ai/docs/pure-word-matching-pure-vector-search-or-combination-of-both-2"
        self.chunk_search_doc =  "https://docs.relevance.ai/docs/better-text-search-chunk-search"
        self.multistep_chunk_doc = "https://docs.relevance.ai/docs/fine-grained-search-search-on-chunks-of-text-data"
        self.advanced_chunk_doc = "https://docs.relevance.ai/docs/fine-grained-search-search-on-chunks-of-text-data-1"
        self.advanced_multistep_chunk_doc = "https://docs.relevance.ai/docs/fine-grained-search-search-on-chunks-of-text-data-2"

        self.initiative_messages = "What else to experiment with :)\n"
        self.category_initiative_messages = {
            'chunk': "if you are searching on large pieces of text, you could chunk your data and try\n", 
            'vector': "if you are looking for strong conceptual relations and not just word matching, you could try\n", 
            'diversity': "if you are looking for strong conceptual relations as well as diverse results, you could try\n", 
            'traditional': "if you are looking for specific text such as id, names, etc., you could try\n", 
        }
    
    def _make_suggestion(self):
        suggestion = self.initiative_messages
        if 'traditional' in self.categories and self.last_search != 'traditional':
            suggestion += self.category_initiative_messages['traditional']
            suggestion += f"   * traditional search ({self.traditional_search_doc})\n"

        if 'vector' in self.categories:
            suggestion += self.category_initiative_messages['vector']
            if self.last_search != "vector":
                 suggestion += f"   * vector search ({self.vector_search_doc})\n"
            if self.last_search != "hybrid":
                 suggestion += f"   * hybrid search ({self.hybrid_search_doc})\n"
            if self.last_search != "semantic":
                 suggestion += f"   * semantic search ({self.semantic_search_doc})\n"
        
        if 'diversity' in self.categories and self.last_search != 'diversity':
            suggestion += self.category_initiative_messages['diversity']
            suggestion += f"   * diversity search ({self.diversity_search_doc})\n"
         
        if 'chunk' in self.categories:
          suggestion += self.category_initiative_messages['chunk']
          if self.last_search != "chunk":
              suggestion += f"   * chunk search ({self.chunk_search_doc})\n"
          if self.last_search != "multistep_chunk":
              suggestion += f"   * multistep_chunk search ({self.multistep_chunk_doc})\n"
          if self.last_search != "advanced_chunk": 
              suggestion += f"   * advanced_chunk search ({self.advanced_chunk_doc})\n"
          if self.last_search != "advanced_multistep_chunk":
              suggestion += f"   * advanced_multistep_chunk search ({self.advanced_multistep_chunk_doc})\n"

        return suggestion