from abc import ABC, abstractmethod

from typing import Any, List, Union

from relevanceai.utils import DocUtils


class ScalerModelBase(ABC, DocUtils):
    model_name: str
    alias: Union[str, None]

    @staticmethod
    def import_from_string(name):
        """It takes a string, splits it on the period, and then imports the module

        Parameters
        ----------
        name
            The name of the class to import.

        Returns
        -------
            The module object.

        """
        components = name.split(".")
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    def vector_name(self, field: str) -> str:
        """If the alias is a string, then return the alias if it contains the string "_vector_", otherwise
        return the alias with "_vector_" appended to it.

        If the alias is not a string, then return "concat_model_name_vector_" if there are more than one
        fields, otherwise return "field_model_name_vector_"

        Parameters
        ----------
        fields : List[str]
            List[str]

        Returns
        -------
            The name of the vector.

        """
        if isinstance(self.alias, str):
            if "_vector_" in self.alias:
                return self.alias
            else:
                return f"{self.alias}_vector_"
        else:

            if "_vector_" in field:
                field_name = field.split("_vector_")[0]

            return f"{field_name}_{self.model_name}_vector_"

    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, *args, **kwargs) -> Any:
        raise NotImplementedError
