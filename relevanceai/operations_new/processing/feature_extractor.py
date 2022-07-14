from collections import defaultdict
from typing import Any, List, Optional, Union

try:
    import torch

    from torch import nn
except:
    pass


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layer_name: Optional[str] = None):
        """`__init__` is a special method in Python classes. It is the method that is called when you
        create an instance of the class

        Parameters
        ----------
        model : nn.Module
            The model to be visualized.
        layer_names : Optional[List[str]]
            A list of layer names to use for the hook. If None, the second last layer is used.

        """

        super().__init__()
        self.model = model

        layer_dict = dict([*self.model.named_modules()])

        if layer_name is None:
            layer_name = list(layer_dict.keys())[-2]

        self.layer_name = layer_name
        self._features: defaultdict = defaultdict(list)

        layer = layer_dict[layer_name]
        layer.register_forward_hook(self.save_outputs_hook(layer_name))

    def save_outputs_hook(self, layer_name):
        """It returns a function that takes 3 arguments (inputs, outputs, and a layer name) and assigns the
        output to the layer name in the _features dictionary

        Parameters
        ----------
        layer_name
            The name of the layer to save the output of.

        Returns
        -------
            A function that takes 3 arguments.

        """

        def fn(_, __, output):
            self._features[layer_name] = output

        return fn

    def forward(self, x: Union[torch.Tensor, List[Any]]) -> List[Any]:
        """The function takes in an input tensor x and passes it through the model. The output of the model
        is not used, but the output of the model's features is returned

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            The features of the model

        """
        if isinstance(x, list):
            x = torch.tensor(x)

        bs = x.shape[0]
        _ = self.model(x)
        x = self._features[self.layer_name]
        x = x.view(bs, -1)
        return x.tolist()


class EmbeddingsExtractor(nn.Module):
    def __init__(self, model: nn.Module):
        """This function takes a model as input and returns a dictionary of all the embedding layers in the
        model

        Parameters
        ----------
        model : nn.Module
            nn.Module

        """
        super().__init__()
        self.model = model

        layer_dict = dict([*self.model.named_modules()])
        self.embedding_layers = {
            k: v for k, v in layer_dict.items() if isinstance(v, nn.Embedding)
        }

    def forward(
        self,
        tokens: Union[torch.Tensor, List[int]],
        embedding_layer: Optional[str] = None,
    ) -> List[List[float]]:
        """It takes a list of tokens, and returns a list of embeddings for those tokens

        Parameters
        ----------
        tokens : List[int]
            List[int]
        embedding_layer : Optional[str]
            The name of the embedding layer to use.

        Returns
        -------
            A list of lists of floats.

        """
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).view(-1)

        if embedding_layer is None:
            embedding_layer = list(self.embedding_layers.keys())[0]

        embeddings = self.embedding_layers[embedding_layer]
        embeddings = embeddings(tokens)

        return embeddings.tolist()
