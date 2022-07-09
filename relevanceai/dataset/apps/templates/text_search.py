import itertools
from relevanceai.dataset.apps.create_apps import CreateApps

class TextSearchTemplate(CreateApps):
    def generate_text_search_config(
        self,
        app_name,
        text_fields,
        text_vector_fields="auto",
        sort=[],
        facets=[],
        return_config_input=False,
    ):
        if text_vector_fields == "auto":
            text_vector_fields = []
            print('Detected "text_vector_fields" is set as "auto", will try to determine "text_vector_fields" from "text_fields"')
            for field, field_type in self.schema.items():
                if isinstance(field_type, dict):
                    for f in text_fields:
                        if f in field:
                            text_vector_fields.append(field)
            print(f'The detected vector fields are {str(text_vector_fields)}')

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

    def create_text_search_app(
        self,
        app_name,
        text_fields,
        text_vector_fields="auto",
        sort=[],
        facets=[],
    ):
        return self.create_app(
            self.generate_text_search_config(
                app_name=app_name, 
                text_fields=text_fields,
                text_vector_fields=text_vector_fields,
                sort=sort,
                facets=facets
            )
        )