import itertools
from relevanceai.dataset.apps.create_apps import CreateApps

class CreateAppsTemplates(CreateApps):
    def create_metrics_chart_app(
        self, 
        app_name="Metrics App", 
        metrics=[], 
        groupby=[], 
        date_fields=[], 
        groupby_depths=[1],
        split_metrics=False, 
        sort=None, 
        page_size=100
    ):
        main_metrics = []
        main_groupby = []
        metric_fields = []
        groupby_fields = []
        metrics_to_sort = []

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
        return self.create_app(config)

    def create_text_search_app(self):
        return 

    def create_text_cluster_app(self):
        return

    def create_image_cluster_app(self):
        return