import pandas as pd
from typing import Callable

class PandasOperations:
    def insert_df(self, dataset_id: str, df: pd.DataFrame, bulk_encode: Callable=None, 
        verbose: bool=True):
        """
        Insert a dataframe
        """
        return self.insert_documents(
            dataset_id=dataset_id, 
            docs=df.to_dict(orient='records'),
            bulk_encode=bulk_encode)
