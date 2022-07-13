import itertools
from relevanceai.dataset.apps.manage_apps import ManageApps

class TextClusterTemplate(ManageApps):
    def generate_text_cluster_config(
        self,
        app_name,
        text_fields,
        text_vector_fields="auto",
        sort=[],
        facets=[],
        return_config_input=False,
    ):
        text_vector_fields = self._auto_detect_vector_fields(
            fields=text_fields, vector_fields=text_vector_fields
        )
        config_input = dict(
            app_name=app_name, 
            default_view="results", 
            search_fields=text_fields,
            preview_fields=text_fields+facets,
            vector_search_fields=text_vector_fields,
            facets=facets,
            sort=sort
        )
        if return_config_input:
            return config_input
        else:
            return self.create_app_config(
                **config_input
            )

    def create_text_cluster_app(
        self,
        app_name,
        text_fields,
        text_vector_fields="auto",
        sort=[],
        facets=[],
    ):
        return self.create_app(
            self.generate_text_cluster_config(
                app_name=app_name, 
                text_fields=text_fields,
                text_vector_fields=text_vector_fields,
                sort=sort,
                facets=facets
            )
        )