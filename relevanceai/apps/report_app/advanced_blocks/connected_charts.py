from relevanceai.apps.report_app.blocks import ReportBlocks

class ConnectedChartsBlocks(ReportBlocks):
    def connected_chart(
        self, 
        groupby=None, 
        metrics=None, 
        sort=None, 
        title:str="",
        page_size:int=20,
        show_frequency: bool = True,
        chart_mode: str = "column",
        timeseries_field: str = None,
        date_interval: str = "monthly",
        add:bool=True
    ):
        sort_by = ""
        sort_direction = "Descending"
        if isinstance(sort, str):
            sort_by = self.dataset._return_sort_in_metrics(sort)
        elif isinstance(sort, dict):
            sort_by = self.dataset._return_sort_in_metrics(list(sort.keys())[0])
            sort_direction = self.dataset._return_sort_in_metrics(list(sort.values())[0])
            if sort_direction == "desc":
                sort_direction = "Descending"
            elif sort_direction == "asc":
                sort_direction = "Ascending"
            elif sort_direction in ["Descending", "Ascending"]:
                pass
            else:
                raise Exception("'sort' dictionary value needs to be of either, 'Ascending' or 'Descending'.")

        groupby = [] if groupby is None else groupby
        metrics = [] if metrics is None else metrics
        
        main_groupby = self.dataset._clean_groupby(groupby)[0]
        main_metrics = self.dataset._clean_metrics(metrics)[0]

        if chart_mode not in ["column", "bar", "line", "scatter", "timeseries"]:
            raise Exception(f"{chart_mode} not supported")

        if chart_mode == "scatter":
            if len(main_metrics) == 1:
                show_frequency = True
            elif len(main_metrics) != 2:
                raise Exception(f"{chart_mode} needs to have 2 metrics specified.")
        elif chart_mode == "timeseries":
            if not timeseries_field:
                raise Exception(f"timeseries_field cannot be unspecified for timeseries chart")

        timeseries_attrs = {}
        if timeseries_field:
            timeseries_attrs = {"field": timeseries_field, "date_interval": date_interval}            


        aggregates = [{**g, "aggType":"groupby"} for g in main_groupby] + [{**m, "aggType":"metric"} for m in main_metrics]
        chart_attrs = {
            "aggregates": aggregates,
            "dataset_id" : self.dataset.dataset_id,
            "displayType" : chart_mode,
            "sortDirection" : sort_direction,
            "title" : title,
            "showFrequencies" : show_frequency,
            "pageSize" : page_size,
            "timeseries" : timeseries_attrs
        }
        chart_attrs["sortBy"] = sort_by
        block = {
            "type": "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "content" : [{
                "type" : "datasetAggregation",
                "attrs" : chart_attrs
            }]
        }
        if add:
            self.contents.append(block)
        return block

    def connected_column_chart(
        self, 
        groupby=None, 
        metrics=None, 
        sort=None, 
        title:str="",
        page_size:int=20,
        show_frequency: bool = True,
        add:bool=True
    ):
        return self.connected_chart(
            groupby=groupby,
            metrics=metrics,
            sort=sort,
            title=title,
            page_size=page_size,
            show_frequency=show_frequency,
            chart_mode="column",
            add=add
        )

    def connected_bar_chart(
        self, 
        groupby=None, 
        metrics=None, 
        sort=None, 
        title:str="",
        page_size:int=20,
        show_frequency: bool = True,
        add:bool=True
    ):
        return self.connected_chart(
            groupby=groupby,
            metrics=metrics,
            sort=sort,
            title=title,
            page_size=page_size,
            show_frequency=show_frequency,
            chart_mode="bar",
            add=add
        )

    def connected_line_chart(
        self, 
        groupby=None, 
        metrics=None, 
        sort=None, 
        title:str="",
        page_size:int=20,
        show_frequency: bool = True,
        add:bool=True
    ):
        return self.connected_chart(
            groupby=groupby,
            metrics=metrics,
            sort=sort,
            title=title,
            page_size=page_size,
            show_frequency=show_frequency,
            chart_mode="line",
            add=add
        )

    def connected_scatter_chart(
        self, 
        groupby=None, 
        metrics=None, 
        sort=None, 
        title:str="",
        page_size:int=20,
        show_frequency: bool = True,
        add:bool=True
    ):
        return self.connected_chart(
            groupby=groupby,
            metrics=metrics,
            sort=sort,
            title=title,
            page_size=page_size,
            show_frequency=show_frequency,
            chart_mode="scatter",
            add=add
        )

    def connected_timeseries_chart(
        self, 
        groupby=None, 
        metrics=None, 
        sort=None, 
        title:str="",
        page_size:int=20,
        show_frequency: bool = True,
        add:bool=True
    ):
        return self.connected_chart(
            groupby=groupby,
            metrics=metrics,
            sort=sort,
            title=title,
            page_size=page_size,
            show_frequency=show_frequency,
            chart_mode="timeseries",
            add=add
        )