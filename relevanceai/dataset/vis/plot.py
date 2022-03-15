from typing import Union

from relevanceai.dataset.statistics.statistics import Statistics


class Plot(Statistics):
    def plot(
        self,
        dr_alias: str,
        number_of_documents: Union[None, int] = None,
        show_progress_bar: bool = True,
    ):
        schema = self.schema
        if dr_alias not in schema:
            raise ValueError("You must reduce your vectors before you can plot")

        import pandas as pd

        pd.options.plotting.backend = "plotly"

        print("Retrieving documents")
        if number_of_documents is None:
            documents = self.get_all_documents(
                select_fields=[dr_alias], show_progress_bar=show_progress_bar
            )
        else:
            documents = self.get_documents(
                number_of_documents=number_of_documents,
                select_fields=[dr_alias],
            )

        dr_algo = dr_alias.split(".")
        df = pd.DataFrame(documents)

        fig = df.plot()
        fig.show()
