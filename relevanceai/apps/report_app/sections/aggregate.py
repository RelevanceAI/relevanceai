import itertools
from typing import Dict, List, Optional, Union
from relevanceai.apps.report_app.advanced_blocks import ReportAdvancedBlocks


class AggregateSections(ReportAdvancedBlocks):
    def section_overview(
        self,
        metrics: List,
        entity_name: str = "documents",
        filters: List = None,
        decimals: int = 3,
    ):
        filters = [] if filters is None else filters

        main_metrics, metric_fields, _ = self.dataset._clean_metrics(metrics)
        self.quote(
            f"Analyzing {self.dataset.shape[0]} {entity_name} across {self.dataset.shape[1]} fields/columns"
        )
        self.h1("Section: Overview")
        overall_metrics = self.dataset.aggregate(metrics=main_metrics, filters=filters)[
            "results"
        ][0]
        overall_metrics_bullets = []
        for k, v in overall_metrics.items():
            if k == "frequency":
                continue
            overall_metrics_bullets += [
                [self.bold(k), ": ", self.bold(str(round(v, decimals)))]
            ]
        self.bullet_list(overall_metrics_bullets)

    def section_groupby(
        self,
        groupby: List,
        metrics: List = None,
        groupby_depths: List = None,
        filters: List = None,
        decimals: int = 3,
    ):
        metrics = [] if metrics is None else metrics
        groupby_depths = [1] if groupby_depths is None else groupby_depths
        filters = [] if filters is None else filters

        self.h1("Section: Analyzing different groups")
        main_metrics, metric_fields, _ = self.dataset._clean_metrics(metrics)
        main_groupby, groupby_fields = self.dataset._clean_groupby(groupby)
        for depth in groupby_depths:
            if depth == 1:
                for group in main_groupby:
                    self.h2(f"Looking at ({group['name']})")
                    for order in ["desc", "asc"]:
                        freq_group = self.dataset.aggregate(
                            groupby=[group],
                            metrics=main_metrics,
                            sort=[{"frequency": order}],
                            filters=filters,
                        )["results"]
                        prefix = "Least" if order == "asc" else "Most"
                        freq_str = f"{prefix} frequent {group['name']}: {freq_group[0]['frequency']} for {freq_group[0][group['name']]}"
                        self.paragraph(freq_str)
                    for metric in main_metrics:
                        for order in ["asc", "desc"]:
                            metric_group = self.dataset.aggregate(
                                groupby=[group],
                                metrics=main_metrics,
                                sort=[{metric["name"]: order}],
                                filters=filters,
                            )["results"]
                            prefix = "Lowest" if order == "asc" else "Highest"
                            group_str = f"{prefix} {metric['name']} for {group['name']}: {round(metric_group[0][metric['name']], decimals)} for {metric_group[0][group['name']]}"
                            self.paragraph(group_str)
            elif depth > 1:
                for group in list(itertools.combinations(main_groupby, depth)):
                    group_name_str = " & ".join([g["name"] for g in group])
                    self.h2(f"Looking at ({group_name_str})")

                    for order in ["desc", "asc"]:
                        freq_group = self.dataset.aggregate(
                            groupby=list(group),
                            metrics=main_metrics,
                            sort=[{"frequency": order}],
                            filters=filters,
                        )["results"]
                        freq_group_name = []
                        for g in group:
                            freq_group_name.append(freq_group[0][g["name"]])
                        freq_group_name_str = " & ".join(freq_group_name)
                        prefix = "Least" if order == "asc" else "Most"
                        freq_group_str = f"{prefix} frequent for {group_name_str}: {freq_group[0]['frequency']} for {freq_group_name_str}"
                        self.paragraph(freq_group_str)
                    for m in main_metrics:
                        for order in ["desc", "asc"]:
                            metric_group = self.dataset.aggregate(
                                groupby=list(group),
                                metrics=main_metrics,
                                sort=[{m["name"]: order}],
                                filters=filters,
                            )["results"]
                            metric_group_name = []
                            for g in group:
                                metric_group_name.append(metric_group[0][g["name"]])
                            metric_group_name_str = " & ".join(metric_group_name)
                            prefix = "Lowest" if order == "asc" else "Highest"
                            group_str = f"{prefix} {m['name']} for {group_name_str}: {round(metric_group[0][m['name']], decimals)} for {metric_group_name_str}"
                            self.paragraph(group_str)


#
