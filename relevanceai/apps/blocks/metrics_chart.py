# import itertools

# class MetricCharts:
#     def metric_block(
#         self,
#         metrics=[],
#         groupby=[],
#         date_fields=[],
#         groupby_depths=[1],
#         split_metrics=False,
#         sort=None,
#         page_size=50,
#         show_frequency=True,
#         return_inputs=False
#     ):
#         if not split_metrics and not metrics:
#             raise TypeError("use 'split_metrics'=False if metrics is empty")
#         main_metrics, metric_fields, _ = self.dataset._clean_metrics(metrics)
#         main_groupby, groupby_fields = self.dataset._clean_groupby(groupby)

#         if sort:
#             for m in main_metrics:
#                 if sort == m['field']:
#                     sort_default = m["name"]

#         charts = []
#         for depth in groupby_depths:
#             if depth == 1:
#                 for group in main_groupby:
#                     if split_metrics:
#                         for metric in main_metrics:
#                             charts.append(
#                                 {
#                                     "groupby": [group],
#                                     "metrics": [metric],
#                                     "sort" : metric['name'],
#                                     "page_size" : page_size,
#                                     "show_frequency" : show_frequency,
#                                     'series-visibility': {metric['name']: True, 'frequency': False}
#                                 }
#                             )
#                     else:
#                         charts.append(
#                             {
#                                 "groupby": [group],
#                                 "metrics": main_metrics,
#                                 "sort" : sort_default,
#                                 "page_size" : page_size,
#                                 "show_frequency" : show_frequency,
#                             }
#                         )
#                     for d in date_fields:
#                         charts.append(
#                             {
#                                 "groupby": [group],
#                                 "page_size" : page_size,
#                                 "show_frequency" : show_frequency,
#                                 "chart_mode" : "timeseries",
#                                 "timeseries" : {"field":d, "date_interval":"monthly"}
#                             }
#                         )
#                         for metric in main_metrics:
#                             charts.append(
#                                 {
#                                     "groupby": [group],
#                                     "metrics": [metric],
#                                     "sort" : metric['name'],
#                                     "page_size" : page_size,
#                                     "show_frequency" : show_frequency,
#                                     "chart_mode" : "timeseries",
#                                     "timeseries" : {"field":d, "date_interval":"monthly"}
#                                 }
#                             )
#             elif depth > 1:
#                 for group in list(itertools.combinations(main_groupby, depth)):
#                     if split_metrics:
#                         for metric in main_metrics:
#                             charts.append(
#                                 {
#                                     "groupby": list(group),
#                                     "metrics": [metric],
#                                     "sort" : metric['name'],
#                                     "page_size" : page_size,
#                                     "show_frequency" : show_frequency,
#                                 }
#                             )
#                     else:
#                         charts.append(
#                             {
#                                 "groupby": list(group),
#                                 "metrics": main_metrics,
#                                 "sort" : sort_default,
#                                 "page_size" : page_size,
#                                 "show_frequency" : show_frequency,
#                             }
#                         )
#                     for d in date_fields:
#                         charts.append(
#                             {
#                                 "groupby": list(group),
#                                 "page_size" : page_size,
#                                 "show_frequency" : show_frequency,
#                                 "chart_mode" : "timeseries",
#                                 "timeseries" : {"field":d, "date_interval":"monthly"}
#                             }
#                         )
#                         for metric in main_metrics:
#                             charts.append(
#                                 {
#                                     "groupby": list(group),
#                                     "metrics": [metric],
#                                     "sort" : metric['name'],
#                                     "page_size" : page_size,
#                                     "show_frequency" : show_frequency,
#                                     "chart_mode" : "timeseries",
#                                     "timeseries" : {"field":d, "date_interval":"monthly"}
#                                 }
#                             )

#         config_inputs = dict(
#             app_name=app_name,
#             default_view="charts",
#             sort_default=sort_default,
#             charts=charts,
#             preview_fields=groupby_fields+metric_fields,
#             facets=groupby_fields,
#             sort=main_metrics
#         )

#         if return_config_input:
#             config_inputs
#         else:
#             return self.create_app_config(
#                 **config_inputs
#             )
