import itertools
from relevanceai.dataset.apps.create_apps import CreateApps

class CreateAppsTemplates(CreateApps):
    """
    Generate is used for templates.
    """
    # def transform_deployable_to_template():
    #     #take deployable as a long string and replace
    #     return

    def generate_metrics_chart_config(
        self, 
        app_name="Metrics Chart App", 
        metrics=[], 
        groupby=[], 
        date_fields=[], 
        groupby_depths=[1],
        split_metrics=False, 
        sort=None, 
        page_size=50
    ):
        main_metrics = []
        main_groupby = []
        metric_fields = []
        groupby_fields = []
        metrics_to_sort = []
        sort_default = None

        if not split_metrics and not metrics:
            raise TypeError("use 'split_metrics'=False if metrics is empty")

        for m in metrics:
            if isinstance(m, str):
                main_metrics.append({"agg" : "avg", "field": m, "name" : f"Average {m}"})
                metric_fields.append(m)
                metrics_to_sort.append(f"Average {m}")
            else:
                main_metrics.append(m)
                metric_fields.append(m['field'])
                metrics_to_sort.append(m['name'])

        for m in groupby:
            if isinstance(m, str):
                if self.schema[m] == "text":
                    main_groupby.append({"agg" : "category", "field": m, "name" : f"{m}"})
                elif self.schema[m] == "numeric":
                    main_groupby.append({"agg" : "numeric", "field": m, "name" : f"{m}"})
                groupby_fields.append(m)
            else:
                main_groupby.append(m)
                groupby_fields.append(m['field'])

        if sort:
            for m in main_metrics:
                if sort == m['field']:
                    sort_default = m["name"]

        charts = []
        for depth in groupby_depths:
            if depth == 1:
                for group in main_groupby:
                    if split_metrics:
                        for metric in main_metrics:
                            charts.append(
                                {
                                    "groupby": [group], 
                                    "metrics": [metric],
                                    "sort" : metric['name'],
                                    "page_size" : page_size
                                }
                            )
                    else:
                        charts.append(
                            {
                                "groupby": [group], 
                                "metrics": main_metrics,
                                "sort" : sort_default,
                                "page_size" : page_size
                            }
                        )
            elif depth > 1:
                for group in list(itertools.combinations(main_groupby, depth)):
                    if split_metrics:
                        for metric in main_metrics:
                            charts.append(
                                {
                                    "groupby": list(group), 
                                    "metrics": [metric],
                                    "sort" : metric['name'],
                                    "page_size" : page_size
                                }
                            )
                    else:
                        charts.append(
                        {
                            "groupby": list(group), 
                            "metrics": main_metrics,
                            "sort" : sort_default,
                            "page_size" : page_size
                        }
                    )

        config = self.create_app_config(
            app_name=app_name, 
            default_view="charts", 
            sort_default=sort_default, 
            charts=charts,
            preview_fields=groupby_fields+metric_fields,
            facets=groupby_fields,
            sort=main_metrics
        )
        return config

    def create_metrics_chart_app(
        self, 
        app_name="Metrics Chart App", 
        metrics=[], 
        groupby=[], 
        date_fields=[], 
        groupby_depths=[1],
        split_metrics=False, 
        sort=None, 
        page_size=50
    ):
        return self.create_app(
            self.generate_metrics_chart_config(
                app_name=app_name, 
                metrics=metrics, 
                groupby=groupby,
                date_fields=date_fields,
                groupby_depths=groupby_depths,
                split_metrics=split_metrics,
                sort=sort,
                page_size=page_size
            )
        )

    def generate_text_search_config(
        self,
        app_name,
        text_fields,
        text_vector_fields="auto",
        sort=[],
        facets=[],
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

        config = self.create_app_config(
            app_name=app_name, 
            default_view="results", 
            search_fields=text_fields,
            preview_fields=text_fields+facets,
            vector_search_fields=text_vector_fields,
            facets=facets,
            sort=sort
        )
        return config

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

    # def create_text_cluster_app(
    #     self,
    #     text_fields,
    #     text_vector_fields,
    # ):
    #     return

    # def create_image_cluster_app(self):
    #     return