import itertools
from relevanceai.dataset.apps.create_apps import CreateApps

class MetricsReportTemplate(CreateApps):
    def generate_metrics_report_config(
        self, 
        app_name:str="Metrics Report", 
        metrics=[], 
        groupby=[], 
        date_fields=[], 
        groupby_depths=[1],
        page_size=50,
        return_config_input=False,
        entity_name="documents",
        decimal_place=3
    ):
        main_metrics, metric_fields, _ = self._clean_metrics(metrics)
        main_groupby, groupby_fields = self._clean_groupby(groupby)
        page_contents = [
            {"content_type": "h1", "content": app_name},
            {"content_type": "quote", "content": f"Analyzing {self.shape[0]} {entity_name} across {self.shape[1]} fields/columns"},
        ]
        page_contents.append({
            "content_type": "h2", "content": "Overall"
        })
        overall_metrics = self.aggregate(metrics=main_metrics)['results'][0]
        page_contents.append({
            "content_type": "bullet", "content": [f"{k}: {v}" for k, v in overall_metrics.items()]
        })
        # page_contents.append({
        #     "content_type": "h2", "content": f"This Month Performance"
        # })
        # for d in date_fields:
        #     results = self.aggregate(
        #         groupby=[{"field":d, "date_interval":"monthly", "agg":"category"}],
        #     )

        for depth in groupby_depths:
            if depth == 1:
                for group in main_groupby:
                    page_contents.append({
                        "content_type": "h2", "content": f"Looking at ({group['name']})"
                    })
                    most_freq_group = self.aggregate(groupby=[group], metrics=main_metrics)['results']
                    page_contents.append({
                        "content_type": "paragraph", "content": f"Most frequent {group['name']}: {most_freq_group[0]['frequency']} for {most_freq_group[0][group['name']]}"
                    })

                    least_freq_group = self.aggregate(groupby=[group], metrics=main_metrics, sort=[{"frequency":"asc"}])['results']
                    page_contents.append({
                        "content_type": "paragraph", "content": f"Least frequent {group['name']}: {least_freq_group[0]['frequency']} for {least_freq_group[0][group['name']]}"
                    })
                    for m in main_metrics:
                        high_group = self.aggregate(groupby=[group], metrics=main_metrics, sort=[{m['name']:"desc"}])['results']
                        page_contents.append({
                            "content_type": "paragraph", "content": f"Highest {m['name']} {group['name']}: {round(high_group[0][m['name']], decimal_place)} for {high_group[0][group['name']]}"
                        })

                        low_group = self.aggregate(groupby=[group], metrics=main_metrics, sort=[{m['name']:"asc"}])['results']
                        page_contents.append({
                            "content_type": "paragraph", "content": f"Lowest {m['name']} {group['name']}: {round(low_group[0][m['name']], decimal_place)} for {low_group[0][group['name']]}"
                        })
            elif depth > 1:
                for group in list(itertools.combinations(main_groupby, depth)):
                    group_name = ' & '.join([g['name'] for g in group])
                    page_contents.append({
                        "content_type": "h2", "content": f"Looking at ({group_name})"
                    })

                    most_freq_group = self.aggregate(groupby=list(group), metrics=main_metrics)['results']
                    most_freq_group_name = []
                    for g in group:
                        most_freq_group_name.append(most_freq_group[0][g['name']])
                    most_freq_group_name = " & ".join(most_freq_group_name)
                    page_contents.append({
                        "content_type": "paragraph", "content": f"Most frequent {group_name}: {most_freq_group[0]['frequency']} for {most_freq_group_name}"
                    })

                    least_freq_group = self.aggregate(groupby=list(group), metrics=main_metrics, sort=[{"frequency":"asc"}])['results']
                    least_freq_group_name = []
                    for g in group:
                        least_freq_group_name.append(least_freq_group[0][g['name']])
                    least_freq_group_name = " & ".join(least_freq_group_name)
                    page_contents.append({
                        "content_type": "paragraph", "content": f"Least frequent {group_name}: {least_freq_group[0]['frequency']} for {least_freq_group_name}"
                    })
                    for m in main_metrics:
                        high_group = self.aggregate(groupby=list(group), metrics=main_metrics, sort=[{m['name']:"desc"}])['results']
                        high_group_name = []
                        for g in group:
                            high_group_name.append(high_group[0][g['name']])
                        high_group_name = " & ".join(high_group_name)
                        page_contents.append({
                            "content_type": "paragraph", "content": f"Highest {m['name']} {group_name}: {round(high_group[0][m['name']], decimal_place)} for {high_group_name}"
                        })

                        low_group = self.aggregate(groupby=list(group), metrics=main_metrics, sort=[{m['name']:"asc"}])['results']
                        low_group_name = []
                        for g in group:
                            low_group_name.append(low_group[0][g['name']])
                        low_group_name = " & ".join(low_group_name)
                        page_contents.append({
                            "content_type": "paragraph", "content": f"Lowest {m['name']} {group_name}: {round(low_group[0][m['name']], decimal_place)} for {low_group_name}"
                        })

        if return_config_input:
            return page_contents
        else:
            return self.create_report_app_config(
                app_name=app_name,
                page_contents=page_contents
            )

    def create_metrics_report_app(
        self, 
        app_name:str="Metrics Report", 
        metrics=[], 
        groupby=[], 
        date_fields=[], 
        groupby_depths=[1],
        page_size=50,
        return_config_input=False,
        entity_name="documents"
    ):
        return self.create_app(
            self.generate_metrics_report_config(
                app_name=app_name, 
                metrics=metrics, 
                groupby=groupby,
                date_fields=date_fields,
                groupby_depths=groupby_depths,
                page_size=page_size,
                entity_name=entity_name
            )
        )