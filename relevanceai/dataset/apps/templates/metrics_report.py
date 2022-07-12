import itertools
from relevanceai.dataset.apps.create_apps import CreateApps
from relevanceai.dataset.apps.report_app import ReportApp

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
        report_app = ReportApp(name=app_name)
        report_app.h1(app_name)
        report_app.quote(f"Analyzing {self.shape[0]} {entity_name} across {self.shape[1]} fields/columns")
        report_app.h2("Overall")
        overall_metrics = self.aggregate(metrics=main_metrics)['results'][0]
        report_app.bullet_list([f"{k}: {v}" for k, v in overall_metrics.items()])
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
                    report_app.h2(f"Looking at ({group['name']})")
                    most_freq_group = self.aggregate(groupby=[group], metrics=main_metrics)['results']
                    most_freq_str = f"Most frequent {group['name']}: {most_freq_group[0]['frequency']} for {most_freq_group[0][group['name']]}"
                    report_app.paragraph(most_freq_str)

                    least_freq_group = self.aggregate(groupby=[group], metrics=main_metrics, sort=[{"frequency":"asc"}])['results']
                    least_freq_str = f"Least frequent {group['name']}: {least_freq_group[0]['frequency']} for {least_freq_group[0][group['name']]}"
                    report_app.paragraph(least_freq_str)
                    for m in main_metrics:
                        high_group = self.aggregate(groupby=[group], metrics=main_metrics, sort=[{m['name']:"desc"}])['results']
                        high_group_str = f"Highest {m['name']} {group['name']}: {round(high_group[0][m['name']], decimal_place)} for {high_group[0][group['name']]}"
                        report_app.paragraph(high_group_str)

                        low_group = self.aggregate(groupby=[group], metrics=main_metrics, sort=[{m['name']:"asc"}])['results']
                        low_group_str = f"Lowest {m['name']} {group['name']}: {round(low_group[0][m['name']], decimal_place)} for {low_group[0][group['name']]}"
                        report_app.paragraph(low_group_str)
            elif depth > 1:
                for group in list(itertools.combinations(main_groupby, depth)):
                    group_name = ' & '.join([g['name'] for g in group])
                    report_app.h2(f"Looking at ({group_name})")

                    most_freq_group = self.aggregate(groupby=list(group), metrics=main_metrics)['results']
                    most_freq_group_name = []
                    for g in group:
                        most_freq_group_name.append(most_freq_group[0][g['name']])
                    most_freq_group_name = " & ".join(most_freq_group_name)
                    most_freq_group_str = f"Most frequent {group_name}: {most_freq_group[0]['frequency']} for {most_freq_group_name}"
                    report_app.paragraph(most_freq_group_str)

                    least_freq_group = self.aggregate(groupby=list(group), metrics=main_metrics, sort=[{"frequency":"asc"}])['results']
                    least_freq_group_name = []
                    for g in group:
                        least_freq_group_name.append(least_freq_group[0][g['name']])
                    least_freq_group_name = " & ".join(least_freq_group_name)
                    least_freq_group_str = f"Least frequent {group_name}: {least_freq_group[0]['frequency']} for {least_freq_group_name}"
                    report_app.paragraph(least_freq_group_str)
                    for m in main_metrics:
                        high_group = self.aggregate(groupby=list(group), metrics=main_metrics, sort=[{m['name']:"desc"}])['results']
                        high_group_name = []
                        for g in group:
                            high_group_name.append(high_group[0][g['name']])
                        high_group_name = " & ".join(high_group_name)
                        high_group_str = f"Highest {m['name']} {group_name}: {round(high_group[0][m['name']], decimal_place)} for {high_group_name}"
                        report_app.paragraph(high_group_str)

                        low_group = self.aggregate(groupby=list(group), metrics=main_metrics, sort=[{m['name']:"asc"}])['results']
                        low_group_name = []
                        for g in group:
                            low_group_name.append(low_group[0][g['name']])
                        low_group_name = " & ".join(low_group_name)
                        low_group_str = f"Lowest {m['name']} {group_name}: {round(low_group[0][m['name']], decimal_place)} for {low_group_name}"
                        report_app.paragraph(low_group_str)

        if return_config_input:
            return report_app.app
        else:
            return self.create_report_app_config(
                report_app
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