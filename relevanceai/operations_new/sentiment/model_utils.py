from typing import List, Set, Tuple
from dataclasses import dataclass

from relevanceai.constants import MissingPackageError

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import (
        pad_sequence,
        PackedSequence,
        pack_padded_sequence,
        pad_packed_sequence,
    )
except ModuleNotFoundError:
    raise MissingPackageError("torch")


@dataclass(eq=True, frozen=True)
class PaddedSequence:
    """A utility class for padding variable length sequences mean for RNN input
    This class is in the style of PackedSequence from the PyTorch RNN Utils,
    but is somewhat more manual in approach. It provides the ability to generate masks
    for outputs of the same input dimensions.
    The constructor should never be called directly and should only be called via
    the autopad classmethod.
    We'd love to delete this, but we pad_sequence, pack_padded_sequence, and
    pad_packed_sequence all require shuffling around tuples of information, and some
    convenience methods using these are nice to have.
    """

    data: torch.Tensor
    batch_sizes: torch.Tensor
    batch_first: bool = False

    @classmethod
    def autopad(
        cls, data, batch_first: bool = False, padding_value=0, device=None
    ) -> "PaddedSequence":
        # handle tensors of size 0 (single item)
        data_ = []
        for d in data:
            if len(d.size()) == 0:
                d = d.unsqueeze(0)
            data_.append(d)
        padded = pad_sequence(
            data_, batch_first=batch_first, padding_value=padding_value
        )
        if batch_first:
            batch_lengths = torch.LongTensor([len(x) for x in data_])
            if any([x == 0 for x in batch_lengths]):
                raise ValueError(
                    "Found a 0 length batch element, this can't possibly be right: {}".format(
                        batch_lengths
                    )
                )
        else:
            # TODO actually test this codepath
            batch_lengths = torch.LongTensor([len(x) for x in data])
        return PaddedSequence(padded, batch_lengths, batch_first).to(device=device)

    def pack_other(self, data: torch.Tensor):
        return pack_padded_sequence(
            data, self.batch_sizes, batch_first=self.batch_first, enforce_sorted=False
        )

    @classmethod
    def from_packed_sequence(
        cls, ps: PackedSequence, batch_first: bool, padding_value=0
    ) -> "PaddedSequence":
        padded, batch_sizes = pad_packed_sequence(ps, batch_first, padding_value)
        return PaddedSequence(padded, batch_sizes, batch_first)

    def cuda(self) -> "PaddedSequence":
        return PaddedSequence(
            self.data.cuda(), self.batch_sizes.cuda(), batch_first=self.batch_first
        )

    def to(
        self, dtype=None, device=None, copy=False, non_blocking=False
    ) -> "PaddedSequence":
        # TODO make to() support all of the torch.Tensor to() variants
        return PaddedSequence(
            self.data.to(
                dtype=dtype, device=device, copy=copy, non_blocking=non_blocking
            ),
            self.batch_sizes.to(device=device, copy=copy, non_blocking=non_blocking),
            batch_first=self.batch_first,
        )

    def mask(
        self, on=int(0), off=int(0), device="cpu", size=None, dtype=None
    ) -> torch.Tensor:
        if size is None:
            size = self.data.size()
        out_tensor = torch.zeros(*size, dtype=dtype)
        # TODO this can be done more efficiently
        out_tensor.fill_(off)
        # note to self: these are probably less efficient than explicilty populating the off values instead of the on values.
        if self.batch_first:
            for i, bl in enumerate(self.batch_sizes):
                out_tensor[i, :bl] = on
        else:
            for i, bl in enumerate(self.batch_sizes):
                out_tensor[:bl, i] = on
        return out_tensor.to(device)

    def unpad(self, other: torch.Tensor) -> List[torch.Tensor]:
        out = []
        for o, bl in zip(other, self.batch_sizes):
            out.append(o[:bl])
        return out

    def flip(self) -> "PaddedSequence":
        return PaddedSequence(
            self.data.transpose(0, 1),
            not self.batch_first,
            self.padding_value,  # type: ignore
        )


def prune_linear_layer(
    layer: nn.Linear,
    index: torch.LongTensor,
    dim: int = 0,
) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.
    Used to remove heads.
    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.
    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(
        layer.weight.device
    )
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def find_pruneable_heads_and_indices(
    heads: List[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: Set[int],
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.
    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.
    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = (
        set(heads) - already_pruned_heads  # type: ignore
    )  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index  # type: ignore
