from relevanceai.apps.report_app.blocks import ReportBlocks

class ConnectedChartsBlocks(ReportBlocks):
    def connected_chart(self, groupby, metrics, sort, title:str="", page_size:int=20, show_frequency:bool=False, add:bool=True):
        block = {
            "type": "appBlock",
            # "attrs" : {"id": str(uuid.uuid4())},
            "attrs" : {
                "aggregates":[
                        {**g, "aggType":"groupby"}
                        for g in groupby
                    ] + [
                        {**m, "aggType":"metric"}
                        for m in metrics
                ],
                "dataset_id" : self.dataset.dataset_id,
                "displayType" : "column",
                "sortBy" : sort,
                "sortDirection" : "Descending",
                "title" : title,
                "showFrequencies" : show_frequency,
                "pageSize" : page_size,
                "timeseries" : {}
            }
        }
        if add:
            self.contents.append(block)
        return block