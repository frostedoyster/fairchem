"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import BackboneInterface, HeadInterface

if TYPE_CHECKING:
    from ase import Atoms

    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from fairchem.core.units.mlip_unit.mlip_unit import Task


AVAILABLE_NORMALIZATIONS = ["LayerNorm", "RMSNorm"]
AVAILABLE_TRANSFORMER_TYPES = ["PostLN", "PreLN"]
AVAILABLE_ACTIVATIONS = ["SiLU", "SwiGLU"]


class Linear(torch.nn.Module):

    def __init__(self, n_feat_in, n_feat_out, scale_factor=1.0):
        super().__init__()
        self.linear_layer = torch.nn.Linear(n_feat_in, n_feat_out)
        self.n_feat_in = n_feat_in if n_feat_in > 0 else 1
        self.linear_layer.weight.data.normal_(0.0, (scale_factor * self.n_feat_in) ** (-0.5))
        self.linear_layer.bias.data.zero_()

    def forward(self, x):
        return self.linear_layer(x)


class DummyModule(torch.nn.Module):
    """Dummy torch module to make torchscript happy.
    This model should never be run"""

    def __init__(self) -> None:
        super(DummyModule, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("This model should never be run")



class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, activation: str) -> None:
        super().__init__()

        # Check if activation is "swiglu" string
        if activation.lower() == "swiglu":
            # SwiGLU mode: single projection produces both "value" and "gate"
            self.w_in = Linear(d_model, 2 * dim_feedforward)
            self.w_out = Linear(dim_feedforward, d_model)
            self.activation = torch.nn.Identity()
            self.is_swiglu = True
        else:
            # Standard mode: regular activation function
            self.w_in = Linear(d_model, dim_feedforward)
            self.w_out = Linear(dim_feedforward, d_model)
            self.activation = getattr(F, activation.lower())
            self.is_swiglu = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_swiglu:
            # SwiGLU activation: split into value and gate
            v, g = self.w_in(x).chunk(2, dim=-1)
            x = v * torch.sigmoid(g)
            x = self.w_out(x)
        else:
            # Standard activation
            x = self.w_in(x)
            x = self.activation(x)
            x = self.w_out(x)
        return x


class AttentionBlock(nn.Module):
    """
    Multi-head attention block.

    :param total_dim: The total dimension of the input and output tensors.
    :param num_heads: The number of attention heads.
    :param temperature: An additional scaling factor for attention scores.
           This is combined with the standard scaling by the square root of
           the head dimension.
    :param epsilon: A small value to avoid division by zero.
    """

    def __init__(
        self,
        total_dim: int,
        num_heads: int,
        temperature: float,
        epsilon: float = 1e-15,
    ) -> None:
        super(AttentionBlock, self).__init__()

        self.input_linear = Linear(total_dim, 3 * total_dim)
        self.output_linear = Linear(total_dim, total_dim)

        self.num_heads = num_heads
        self.epsilon = epsilon
        self.temperature = temperature
        if total_dim % num_heads != 0:
            raise ValueError("total dimension is not divisible by the number of heads")
        self.head_dim = total_dim // num_heads

    def forward(
        self, x: torch.Tensor, cutoff_factors: torch.Tensor, use_manual_attention: bool
    ) -> torch.Tensor:
        """
        Forward pass for the attention block.

        :param x: The input tensor, of shape (batch_size, seq_length, total_dim)
        :param cutoff_factors: The cutoff factors for the edges, of shape
            (batch_size, seq_length, seq_length)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: The output tensor, of shape (batch_size, seq_length, total_dim)
        """
        initial_shape = x.shape
        x = self.input_linear(x)
        x = x.reshape(
            initial_shape[0], initial_shape[1], 3, self.num_heads, self.head_dim
        )
        x = x.permute(2, 0, 3, 1, 4)

        queries, keys, values = x[0], x[1], x[2]
        attn_weights = torch.clamp(cutoff_factors[:, None, :, :], self.epsilon)
        attn_weights = torch.log(attn_weights)
        if use_manual_attention:
            x = manual_attention(
                queries, keys, values, attn_weights, self.temperature
            )
        else:
            x = torch.nn.functional.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attn_weights,
                scale=1.0 / (self.head_dim**0.5 * self.temperature),
            )
        x = x.transpose(1, 2).reshape(initial_shape)
        x = self.output_linear(x)
        return x


class TransformerLayer(torch.nn.Module):
    """
    Single layer of a Transformer.

    :param d_model: The dimension of the model.
    :param n_heads: The number of attention heads.
    :param dim_node_features: The dimension of the node features.
    :param dim_feedforward: The dimension of the feedforward network.
    :param norm: The normalization type, either "LayerNorm" or "RMSNorm".
    :param activation: The activation function, either "SiLU" or "SwiGLU".
    :param transformer_type: The type of transformer, either "PostLN" or "PreLN".
    :param temperature: An additional scaling factor for attention scores.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_node_features: int,
        dim_feedforward: int = 512,
        norm: str = "LayerNorm",
        activation: str = "SiLU",
        transformer_type: str = "PostLN",
        temperature: float = 1.0,
    ) -> None:
        super(TransformerLayer, self).__init__()
        self.attention = AttentionBlock(d_model, n_heads, temperature)
        self.transformer_type = transformer_type
        self.d_model = d_model
        norm_class = getattr(nn, norm)
        self.norm_attention = norm_class(d_model)
        self.norm_mlp = norm_class(d_model)
        self.mlp = FeedForward(d_model, dim_feedforward, activation)
        self.expanded_node_features = False
        if dim_node_features != d_model:
            self.expanded_node_features = True
            self.center_contraction = Linear(dim_node_features, d_model)
            self.center_expansion = Linear(d_model, dim_node_features)
            self.norm_center_features = norm_class(dim_node_features)
            self.center_mlp = FeedForward(
                dim_node_features, 2 * dim_node_features, activation
            )
        else:
            self.center_contraction = torch.nn.Identity()
            self.center_expansion = torch.nn.Identity()
            self.norm_center_features = torch.nn.Identity()
            self.center_mlp = torch.nn.Identity()

    def _forward_pre_ln_impl(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("before contraction", node_embeddings.mean().item(), node_embeddings.std().item(), flush=True)

        if self.expanded_node_features:
            input_node_embeddings = self.center_contraction(node_embeddings)
        else:
            input_node_embeddings = node_embeddings

        # print("before expanded", input_node_embeddings.mean().item(), input_node_embeddings.std().item(), flush=True)

        tokens = torch.cat([input_node_embeddings, edge_embeddings], dim=1)
        new_tokens = self.attention(
            self.norm_attention(tokens), cutoff_factors, use_manual_attention
        )
        output_node_embeddings, output_edge_embeddings = torch.split(
            new_tokens, [1, new_tokens.shape[1] - 1], dim=1
        )
        if self.expanded_node_features:
            # print("expanded node features", flush=True)
            output_node_embeddings = node_embeddings + self.center_expansion(
                output_node_embeddings
            )
            output_node_embeddings = output_node_embeddings + self.center_mlp(
                self.norm_center_features(output_node_embeddings)
            )

        output_edge_embeddings = edge_embeddings + output_edge_embeddings
        output_edge_embeddings = output_edge_embeddings + self.mlp(
            self.norm_mlp(output_edge_embeddings)
        )

        # print("after expanded", output_node_embeddings.mean().item(), output_node_embeddings.std().item(), flush=True)

        return output_node_embeddings, output_edge_embeddings

    def _forward_post_ln_impl(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.expanded_node_features:
            input_node_embeddings = self.center_contraction(node_embeddings)
        else:
            input_node_embeddings = node_embeddings
        tokens = torch.cat([input_node_embeddings, edge_embeddings], dim=1)
        tokens = self.norm_attention(
            tokens + self.attention(tokens, cutoff_factors, use_manual_attention)
        )
        tokens = self.norm_mlp(tokens + self.mlp(tokens))
        output_node_embeddings, output_edge_embeddings = torch.split(
            tokens, [1, tokens.shape[1] - 1], dim=1
        )
        if self.expanded_node_features:
            output_node_embeddings = node_embeddings + self.center_expansion(
                output_node_embeddings
            )
            output_node_embeddings = output_node_embeddings + self.center_mlp(
                self.norm_center_features(output_node_embeddings)
            )
        return output_node_embeddings, output_edge_embeddings

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single Transformer layer.

        :param node_embeddings: The input node embeddings, of shape
            (batch_size, d_model)
        :param edge_embeddings: The input edge embeddings, of shape
            (batch_size, seq_length, d_model)
        :param cutoff_factors: The cutoff factors for the edges, of shape
            (batch_size, seq_length, seq_length)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: A tuple containing:
            - The output node embeddings, of shape (batch_size, d_model)
            - The output edge embeddings, of shape (batch_size, seq_length, d_model)
        """
        if self.transformer_type == "PostLN":
            node_embeddings, edge_embeddings = self._forward_post_ln_impl(
                node_embeddings,
                edge_embeddings,
                cutoff_factors,
                use_manual_attention,
            )
        if self.transformer_type == "PreLN":
            node_embeddings, edge_embeddings = self._forward_pre_ln_impl(
                node_embeddings,
                edge_embeddings,
                cutoff_factors,
                use_manual_attention,
            )
        return node_embeddings, edge_embeddings


class Transformer(torch.nn.Module):
    """
    Transformer implementation.

    :param d_model: The dimension of the model.
    :param num_layers: The number of transformer layers.
    :param n_heads: The number of attention heads.
    :param dim_node_features: The dimension of the node features.
    :param dim_feedforward: The dimension of the feedforward network.
    :param norm: The normalization type, either "LayerNorm" or "RMSNorm".
    :param activation: The activation function, either "SiLU" or "SwiGLU".
    :param transformer_type: The type of transformer, either "PostLN" or "PreLN".
    :param attention_temperature: The temperature scaling factor for attention
        scores. This is combined with the standard scaling by the square root of
        the head dimension.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        n_heads: int,
        dim_node_features: int,
        dim_feedforward: int = 512,
        norm: str = "LayerNorm",
        activation: str = "SiLU",
        transformer_type: str = "PostLN",
        attention_temperature: float = 1.0,
    ) -> None:
        super(Transformer, self).__init__()
        if norm not in AVAILABLE_NORMALIZATIONS:
            raise ValueError(
                f"Unknown normalization flag: {norm}. "
                f"Please choose from: {AVAILABLE_NORMALIZATIONS}"
            )

        if transformer_type not in AVAILABLE_TRANSFORMER_TYPES:
            raise ValueError(
                f"Unknown transformer flag: {transformer_type}. "
                f"Please choose from: {AVAILABLE_TRANSFORMER_TYPES}"
            )
        self.transformer_type = transformer_type

        if activation not in AVAILABLE_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation flag: {activation}. "
                f"Please choose from: {AVAILABLE_ACTIVATIONS}"
            )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_node_features=dim_node_features,
                    dim_feedforward=dim_feedforward,
                    norm=norm,
                    activation=activation,
                    transformer_type=transformer_type,
                    temperature=attention_temperature,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Transformer.

        :param node_embeddings: The input node embeddings, of shape
            (batch_size, d_model)
        :param edge_embeddings: The input edge embeddings, of shape
            (batch_size, seq_length, d_model)
        :param cutoff_factors: The cutoff factors for the edges, of shape
            (batch_size, seq_length, seq_length)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: A tuple containing:
            - The output node embeddings, of shape (batch_size, d_model)
            - The output edge embeddings, of shape (batch_size, seq_length, d_model)
        """
        for layer in self.layers:
            node_embeddings, edge_embeddings = layer(
                node_embeddings, edge_embeddings, cutoff_factors, use_manual_attention
            )
        return node_embeddings, edge_embeddings


class CartesianTransformer(torch.nn.Module):
    """
    Cartesian Transformer implementation for handling 3D coordinates.

    :param cutoff: The cutoff distance for neighbor interactions.
    :param cutoff_width: The width of the cutoff function.
    :param d_model: The dimension of the model.
    :param n_head: The number of attention heads.
    :param dim_node_features: The dimension of the node features.
    :param dim_feedforward: The dimension of the feedforward network.
    :param n_layers: The number of transformer layers.
    :param norm: The normalization type, either "LayerNorm" or "RMSNorm".
    :param activation: The activation function, either "SiLU" or "SwiGLU".
    :param attention_temperature: The temperature scaling factor for attention scores.
    :param transformer_type: The type of transformer, either "PostLN" or "PreLN".
    :param n_atomic_species: The number of atomic species.
    :param is_first: Whether this is the first transformer in the model.
    """

    def __init__(
        self,
        cutoff: float,
        cutoff_width: float,
        d_model: int,
        n_head: int,
        dim_node_features: int,
        dim_feedforward: int,
        n_layers: int,
        norm: str,
        activation: str,
        attention_temperature: float,
        transformer_type: str,
        n_atomic_species: int,
        is_first: bool,
    ) -> None:
        super(CartesianTransformer, self).__init__()
        self.is_first = is_first
        self.cutoff = cutoff
        self.cutoff_width = cutoff_width
        self.trans = Transformer(
            d_model=d_model,
            num_layers=n_layers,
            n_heads=n_head,
            dim_node_features=dim_node_features,
            dim_feedforward=dim_feedforward,
            norm=norm,
            activation=activation,
            transformer_type=transformer_type,
            attention_temperature=attention_temperature,
        )

        # self.edge_embedder = Linear(4, d_model)

        # if not is_first:
        #     n_merge = 3
        # else:
        #     n_merge = 2

        # self.compress = nn.Sequential(
        #     Linear(n_merge * d_model, d_model),
        #     torch.nn.SiLU(),
        #     Linear(d_model, d_model),
        # )

        # self.neighbor_embedder = DummyModule()  # for torchscript
        # if not is_first:
        #     self.neighbor_embedder = nn.Embedding(n_atomic_species, d_model)

    def forward(
        self,
        input_node_embeddings: torch.Tensor,
        input_messages: torch.Tensor,
        element_indices_neighbors: torch.Tensor,
        edge_vectors: torch.Tensor,
        padding_mask: torch.Tensor,
        edge_distances: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CartesianTransformer.

        :param input_node_embeddings: The input node embeddings, of shape
            (n_nodes, d_model)
        :param input_messages: The input messages to the transformer, of shape
            (n_nodes, max_num_neighbors, d_model)
        :param element_indices_neighbors: The atomic species of the neighboring atoms,
            of shape (n_nodes, max_num_neighbors)
        :param edge_vectors: The cartesian edge vectors between the central atoms and
            their neighbors, of shape (n_nodes, max_num_neighbors, 3)
        :param padding_mask: A padding mask indicating which neighbors are real, and
            which are padded, of shape (n_nodes, max_num_neighbors)
        :param edge_distances: The distances between the central atoms and their
            neighbors, of shape (n_nodes, max_num_neighbors)
        :param cutoff_factors: The cutoff factors for the edges, of shape
            (n_nodes, max_num_neighbors)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: A tuple containing:
            - The output node embeddings, of shape (n_nodes, d_model)
            - The output edge embeddings, of shape (n_nodes, max_num_neighbors, d_model)
        """
        node_embeddings = input_node_embeddings
        # edge_embeddings = [edge_vectors, edge_distances[:, :, None]]

        # on some systems, on isolated atoms, a torchscript bug concatenates the two
        # (empty) float tensors into an int tensors, causing an error later on
        # edge_embeddings = torch.cat(edge_embeddings, dim=2).to(edge_vectors.dtype)

        # edge_embeddings = self.edge_embedder(edge_embeddings)

        # if not self.is_first:
        #     neighbor_elements_embeddings = self.neighbor_embedder(
        #         element_indices_neighbors
        #     )
        #     edge_tokens = torch.cat(
        #         [edge_embeddings, neighbor_elements_embeddings, input_messages], dim=2
        #     )
        # else:
        #     neighbor_elements_embeddings = torch.empty(
        #         0, device=edge_vectors.device, dtype=edge_vectors.dtype
        #     )  # for torch script
        #     edge_tokens = torch.cat([edge_embeddings, input_messages], dim=2)

        edge_tokens = input_messages
        # tokens = torch.cat([node_elements_embedding[:, None, :], tokens], dim=1)

        padding_mask_with_central_token = torch.ones(
            padding_mask.shape[0], dtype=torch.bool, device=padding_mask.device
        )
        total_padding_mask = torch.cat(
            [padding_mask_with_central_token[:, None], padding_mask], dim=1
        )

        cutoff_subfactors = torch.ones(
            padding_mask.shape[0],
            dtype=cutoff_factors.dtype,
            device=padding_mask.device,
        )
        cutoff_factors = torch.cat([cutoff_subfactors[:, None], cutoff_factors], dim=1)
        cutoff_factors[~total_padding_mask] = 0.0

        cutoff_factors = cutoff_factors[:, None, :]
        cutoff_factors = cutoff_factors.repeat(1, cutoff_factors.shape[2], 1)

        initial_num_tokens = edge_vectors.shape[1]

        # print("before attention", node_embeddings.mean().item(), node_embeddings.std().item(), flush=True)

        output_node_embeddings, output_edge_embeddings = self.trans(
            node_embeddings[:, None, :],
            edge_tokens,
            cutoff_factors=cutoff_factors,
            use_manual_attention=use_manual_attention,
        )

        # print("after attention", output_node_embeddings.mean().item(), output_node_embeddings.std().item(), flush=True)

        output_node_embeddings = output_node_embeddings.squeeze(1)
        return output_node_embeddings, output_edge_embeddings


def manual_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Implements the attention operation manually, using basic PyTorch operations.
    We need it because the built-in PyTorch attention does not support double backward,
    which is needed when training with conservative forces.

    :param q: The queries
    :param k: The keys
    :param v: The values
    :param attn_mask: The attention mask
    :param temperature: An additional scaling factor for attention scores.
    :return: The result of the attention operation
    """
    attention_weights = (
        torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5 * temperature)
    ) + attn_mask
    attention_weights = attention_weights.softmax(dim=-1)
    attention_output = torch.matmul(attention_weights, v)
    return attention_output



def cutoff_func_cosine(
    values: torch.Tensor, cutoff: torch.Tensor, width: float
) -> torch.Tensor:
    """
    Cosine cutoff function.

    :param values: Distances at which to evaluate the cutoff function.
    :param cutoff: Cutoff radius for each node.
    :param width: Width of the cutoff region.
    :return: Values of the cutoff function at the specified distances.
    """

    scaled_values = (values - (cutoff - width)) / width

    mask_smaller = scaled_values <= 0.0
    mask_active = (scaled_values > 0.0) & (scaled_values < 1.0)

    f = torch.zeros_like(scaled_values)

    f[mask_active] = 0.5 + 0.5 * torch.cos(torch.pi * scaled_values[mask_active])
    f[mask_smaller] = 1.0
    return f


def get_nef_indices(
    centers: torch.Tensor, n_nodes: int, n_edges_per_node: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes tensors of indices useful to convert between edge
    and NEF layouts; the usage and function of `nef_indices` and
    `nef_to_edges_neighbor` is clear in the ``edge_array_to_nef``
    and ``nef_array_to_edges`` functions below.

    :param centers: A 1D tensor of shape (n_edges,) containing the
        indices of the center nodes for each edge, with the center nodes
        being the "i" node in an "i -> j" edge.
    :param n_nodes: The number of nodes in the graph.
    :param n_edges_per_node: The maximum number of edges per node.

    :return: A tuple with three tensors (nef_indices, nef_to_edges_neighbor, nef_mask).
        In particular:
        nef_array = edge_array[nef_indices]
        edge_array = nef_array[centers, nef_to_edges_neighbor]
        The third output, nef_mask, is a mask that can be used to
        filter out the padding values in the NEF array, as different
        nodes will have, in general, different number of edges.
    """

    bincount = torch.bincount(centers, minlength=n_nodes)

    arange = torch.arange(n_edges_per_node, device=centers.device)
    arange_expanded = arange.view(1, -1).expand(n_nodes, -1)
    nef_mask = arange_expanded < bincount.view(-1, 1)

    argsort = torch.argsort(centers, stable=True)

    nef_indices = torch.zeros(
        (n_nodes, n_edges_per_node), dtype=torch.long, device=centers.device
    )
    nef_indices[nef_mask] = argsort

    return nef_indices, nef_mask


def get_corresponding_edges(array: torch.Tensor) -> torch.Tensor:
    """
    Computes the corresponding edge (i.e., the edge that goes in the
    opposite direction) for each edge in the array; this is useful
    in the message-passing operation.

    :param array: A 2D tensor of shape (n_edges, 5). For each i -> j
        edge, the first column contains the index of the center node i,
        the second column contains the index of the neighbor node j,
        and the last three columns contain the cell shifts along x, y, and z
        directions, respectively.

    :return: A 1D tensor of shape (n_edges,) containing, for each edge,
        the index of the corresponding edge (i.e., the edge that goes
        in the opposite direction). If the input array is empty, an
        empty tensor is returned.
    """

    if array.numel() == 0:
        return torch.empty((0,), dtype=array.dtype, device=array.device)

    array = array.to(torch.int64)  # avoid overflow

    centers = array[:, 0]
    neighbors = array[:, 1]
    cell_shifts_x = array[:, 2]
    cell_shifts_y = array[:, 3]
    cell_shifts_z = array[:, 4]

    # will be useful later
    negative_cell_shifts_x = -cell_shifts_x
    negative_cell_shifts_y = -cell_shifts_y
    negative_cell_shifts_z = -cell_shifts_z

    # create a unique identifier for each edge
    # first, we shift the cell_shifts so that the minimum value is 0
    min_cell_shift_x = cell_shifts_x.min()
    cell_shifts_x = cell_shifts_x - min_cell_shift_x
    negative_cell_shifts_x = negative_cell_shifts_x - min_cell_shift_x

    min_cell_shift_y = cell_shifts_y.min()
    cell_shifts_y = cell_shifts_y - min_cell_shift_y
    negative_cell_shifts_y = negative_cell_shifts_y - min_cell_shift_y

    min_cell_shift_z = cell_shifts_z.min()
    cell_shifts_z = cell_shifts_z - min_cell_shift_z
    negative_cell_shifts_z = negative_cell_shifts_z - min_cell_shift_z

    max_centers_neigbors = centers.max() + 1  # same as neighbors.max() + 1
    max_shift_x = cell_shifts_x.max() + 1
    max_shift_y = cell_shifts_y.max() + 1
    max_shift_z = cell_shifts_z.max() + 1

    size_1 = max_shift_z
    size_2 = max_shift_y * size_1
    size_3 = max_shift_x * size_2
    size_4 = max_centers_neigbors * size_3

    unique_id = (
        centers * size_4
        + neighbors * size_3
        + cell_shifts_x * size_2
        + cell_shifts_y * size_1
        + cell_shifts_z
    )

    # the inverse is the same, but centers and neighbors are swapped
    # and we use the negative values of the cell_shifts
    unique_id_inverse = (
        neighbors * size_4
        + centers * size_3
        + negative_cell_shifts_x * size_2
        + negative_cell_shifts_y * size_1
        + negative_cell_shifts_z
    )

    unique_id_argsort = unique_id.argsort()
    unique_id_inverse_argsort = unique_id_inverse.argsort()

    corresponding_edges = torch.empty_like(centers)
    corresponding_edges[unique_id_argsort] = unique_id_inverse_argsort

    return corresponding_edges.to(array.dtype)


def edge_array_to_nef(
    edge_array: torch.Tensor,
    nef_indices: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Converts an edge array to a NEF array.

    :param edge_array: A tensor where the first dimension is the index of
        the edge, i.e. with shape (n_edges, ...).
    :param nef_indices: The indices to convert from edge to NEF layout,
        as returned by the ``get_nef_indices`` function.
    :param mask: An optional boolean mask of shape (n_nodes, n_edges_per_node),
        as returned by the ``get_nef_indices`` function. If provided,
        the output NEF array will have the values in the positions
        where the mask is False set to ``fill_value``.
    :param fill_value: The value to use to fill the positions in the
        NEF array where the mask is False. Only used if ``mask`` is
        provided.

    :return: A tensor with the same information as ``edge_array``,
        but in NEF layout, i.e. with shape (n_nodes, n_edges_per_node, ...).
        If ``mask`` is provided, the values in the positions where
        the mask is False are set to ``fill_value``.
    """
    if mask is None:
        return edge_array[nef_indices]
    else:
        return torch.where(
            mask.reshape(mask.shape + (1,) * (len(edge_array.shape) - 1)),
            edge_array[nef_indices],
            fill_value,
        )


def nef_array_to_edges(
    nef_array: torch.Tensor, centers: torch.Tensor, nef_to_edges_neighbor: torch.Tensor
) -> torch.Tensor:
    """Converts a NEF array to an edge array.

    :param nef_array: A tensor where the first two dimensions are the
        indices of the NEF layout, i.e. with shape (n_nodes, n_edges_per_node, ...).
    :param centers: The indices of the center nodes for each edge.
    :param nef_to_edges_neighbor: The indices of the edges for each
        neighbor in the NEF layout, as returned by the ``get_nef_indices`` function.

    :return: A tensor with the same information as ``nef_array``,
        but in edge layout, i.e. with shape (n_edges, ...).
    """
    return nef_array[centers, nef_to_edges_neighbor]


def compute_reversed_neighbor_list(
    nef_indices: torch.Tensor,
    corresponding_edges: torch.Tensor,
    nef_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Creates a reversed neighborlist, where for each
    center atom `i` and its neighbor `j` in the original
    neighborlist, the position of atom `i` in the list
    of neighbors of atom `j` is returned.

    :param nef_indices: The indices to convert from edge to NEF layout,
        as returned by the ``get_nef_indices`` function.
    :param corresponding_edges: The indices of the corresponding edges,
        as returned by the ``get_corresponding_edges`` function.
    :param nef_mask: A boolean mask of shape (n_nodes, n_edges_per_node),
        as returned by the ``get_nef_indices`` function.
    :return: A tensor of the same shape as ``nef_indices``,
        where each entry contains the position of the center
        atom in the neighborlist of the corresponding neighbor atom.
    """
    num_atoms, max_num_neighbors = nef_indices.shape

    flat_edge_indices = nef_indices.reshape(-1)
    flat_positions = torch.arange(max_num_neighbors, device=nef_indices.device).repeat(
        num_atoms
    )
    flat_mask = nef_mask.reshape(-1)

    if flat_edge_indices.numel() == 0:
        max_edge_index = 0
    else:
        max_edge_index = int(flat_edge_indices.max().item()) + 1
    size: List[int] = [max_edge_index]

    edge_index_to_position = torch.full(
        size,
        0,
        dtype=torch.long,
        device=nef_indices.device,
    )
    edge_index_to_position[flat_edge_indices[flat_mask]] = flat_positions[flat_mask]

    reverse_edge_idx = corresponding_edges[nef_indices]
    reversed_neighbor_list = edge_index_to_position[reverse_edge_idx]
    reversed_neighbor_list = reversed_neighbor_list.masked_fill(~nef_mask, 0)

    return reversed_neighbor_list


def _batch_index(data: AtomicData) -> torch.Tensor:
    if "batch" in data:
        return data["batch"].long()
    return torch.zeros(
        data["pos"].shape[0], device=data["pos"].device, dtype=torch.long
    )


def _num_graphs(data: AtomicData) -> int:
    if "natoms" in data:
        return int(data["natoms"].numel())
    return 1


def _sum_positions_by_graph(
    pos: torch.Tensor, batch: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    per_atom_position_sum = pos.sum(dim=-1, keepdim=True)
    energy = pos.new_zeros((num_graphs, 1))
    energy.index_add_(0, batch, per_atom_position_sum)
    return energy.squeeze(-1)


@registry.register_model("PET_backbone")
class PETBackbone(nn.Module, BackboneInterface):
    """Dummy PET backbone used to wire PET into the MLIP stack.

    This is intentionally minimal while PET is being integrated. It preserves the
    backbone/head contract used by HydraModel and exposes enough trainable state
    for the normal training path to exercise optimizers, DDP, checkpointing, and
    loss computation.
    """

    def __init__(
        self,
        regress_forces: bool = True,
        direct_forces: bool = True,
        regress_stress: bool = True,
        direct_stress: bool = True,
        dens_enabled: bool = False,
        max_num_elements: int = 1000,
        cutoff: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.regress_stress = regress_stress
        self.direct_stress = direct_stress
        self.dens_enabled = dens_enabled
        self.max_num_elements = max_num_elements
        self.cutoff = cutoff
        self.extra_config = kwargs

        # hardcoded hypers
        self.d_pet = 256
        self.node_to_edge_ratio = 4
        self.feedforward_ratio = 2
        self.num_gnn_layers = 3
        self.num_attention_layers = 1
        self.cutoff_width = 0.5
        self.num_heads = 8
        self.cutoff_function = "cosine"
        self.attention_temperature = 1.0

        self.node_embedder = nn.Embedding(self.max_num_elements, self.d_pet * self.node_to_edge_ratio)
        if self.dens_enabled:
            self.dens_force_encoder = torch.nn.Sequential(
                Linear(4, self.node_to_edge_ratio * self.d_pet),
                torch.nn.SiLU(),
                Linear(self.node_to_edge_ratio * self.d_pet, self.node_to_edge_ratio * self.d_pet),
            )
        self.edge_center_embedder = nn.Embedding(self.max_num_elements, self.d_pet)
        self.edge_neighbor_embedder = nn.Embedding(self.max_num_elements, self.d_pet)
        self.edge_directional_embedder = Linear(4, self.d_pet)
        self.edge_compressor = torch.nn.Sequential(
            Linear(3 * self.d_pet, self.d_pet),
            torch.nn.SiLU(),
            Linear(self.d_pet, self.d_pet),
            torch.nn.SiLU(),
            Linear(self.d_pet, self.d_pet),
            torch.nn.SiLU(),
            Linear(self.d_pet, self.d_pet),

        )

        self.combination_norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(self.feedforward_ratio * self.d_pet) for _ in range(self.num_gnn_layers)]
        )
        self.combination_mlps = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    Linear(self.feedforward_ratio * self.d_pet, self.feedforward_ratio * self.d_pet),
                    torch.nn.SiLU(),
                    Linear(self.feedforward_ratio * self.d_pet, self.d_pet),
                )
                for _ in range(self.num_gnn_layers)
            ]
        )
        self.gnn_layers = torch.nn.ModuleList(
            [
                CartesianTransformer(
                    self.cutoff,
                    self.cutoff_width,
                    self.d_pet,
                    self.num_heads,
                    self.node_to_edge_ratio * self.d_pet,
                    self.feedforward_ratio * self.d_pet,
                    self.num_attention_layers,
                    "RMSNorm",
                    "SwiGLU",
                    self.attention_temperature,
                    "PreLN",
                    self.max_num_elements,
                    layer_index == 0,  # is first layer
                )
                for layer_index in range(self.num_gnn_layers)
            ]
        )
        self.edge_expander = Linear(self.d_pet, self.d_pet * self.node_to_edge_ratio)
        self.edge_expander.linear_layer.weight.data.zero_()


    @classmethod
    def build_inference_settings(cls, settings: InferenceSettings) -> dict:
        return {}

    def get_default_untrained_tasks(
        self,
        checkpoint_tasks: dict[str, Task],
        inference_settings: InferenceSettings,
    ) -> list[Task]:
        return []

    def validate_tasks(self, dataset_to_tasks: dict[str, list]) -> None:
        pass

    def prepare_for_inference(self, data: AtomicData, settings: InferenceSettings):
        return self

    def on_predict_check(self, data: AtomicData) -> None:
        pass

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None:
        pass

    def _forward_impl(self, data: AtomicData) -> dict[str, torch.Tensor]:
        positions = data["pos"]
        atomic_numbers = data["atomic_numbers"]
        cell = data["cell"]
        pbc = data["pbc"]
        edge_index = data["edge_index"]
        neighbors, centers = torch.unbind(edge_index, dim=0)
        cell_offsets = data["cell_offsets"]
        batch = data["batch"]

        displacement = None
        if self.regress_stress and not self.direct_stress:
            displacement = torch.zeros(
                (len(cell), 3, 3),
                dtype=positions.dtype,
                device=positions.device,
            )
            displacement.requires_grad_(True)
            symmetric_displacement = 0.5 * (
                displacement + displacement.transpose(-1, -2)
            )
            positions = positions + torch.bmm(
                positions.unsqueeze(-2),
                torch.index_select(symmetric_displacement, 0, batch),
            ).squeeze(-2)
            cell = cell + torch.bmm(cell, symmetric_displacement)

        if self.regress_forces and not self.direct_forces:
            positions.requires_grad_(True)

        data["pos"] = positions
        data["cell"] = cell

        data["atomic_numbers_full"] = atomic_numbers
        data["batch_full"] = batch

        # somehow the backward of this operation is very slow at evaluation,
        # where there is only one cell, therefore we simplify the calculation
        # for that case
        if len(cell) == 1:
            cell_contributions = cell_offsets @ cell[0]
        else:
            cell_contributions = torch.einsum(
                "ab, abc -> ac",
                cell_offsets,
                cell[batch[centers]],
            )
        edge_vectors = positions[neighbors] - positions[centers] + cell_contributions

        num_atoms = len(positions)

        num_neighbors = torch.bincount(centers, minlength=num_atoms)
        max_edges_per_node = int(torch.max(num_neighbors))

        edge_distances = torch.sqrt(torch.sum(edge_vectors**2, dim=-1))
        if self.cutoff_function.lower() == "cosine":
            cutoff_factors = cutoff_func_cosine(edge_distances, self.cutoff, self.cutoff_width)
        else:
            raise ValueError(
                f"Unknown cutoff function type: {self.cutoff_function}. "
                f"Supported types are 'cosine'."
            )

        # Convert to NEF (Node-Edge-Feature) format:
        nef_indices, nef_mask = get_nef_indices(centers, num_atoms, max_edges_per_node)

        # Element indices
        atomic_numbers_centers = atomic_numbers[centers]
        atomic_numbers_neighbors = atomic_numbers[neighbors]

        # Send everything to NEF:
        edge_vectors = edge_array_to_nef(edge_vectors, nef_indices)
        edge_distances = edge_array_to_nef(edge_distances, nef_indices)

        # TODO: turn into something more sane once you just do it at the beginning
        element_indices_centers = edge_array_to_nef(atomic_numbers_centers, nef_indices)
        element_indices_neighbors = edge_array_to_nef(atomic_numbers_neighbors, nef_indices)
        cutoff_factors = edge_array_to_nef(cutoff_factors, nef_indices, nef_mask, 0.0)

        corresponding_edges = get_corresponding_edges(
            torch.concatenate(
                [centers.unsqueeze(-1), neighbors.unsqueeze(-1), cell_offsets.to(centers.dtype)],
                dim=-1,
            )
        )

        # These are the two arrays we need for message passing with edge reversals,
        # if indexing happens in a two-dimensional way:
        # edges_ji = edges_ij[reversed_neighbor_list, neighbors_index]
        reversed_neighbor_list = compute_reversed_neighbor_list(
            nef_indices, corresponding_edges, nef_mask
        )
        neighbors_index = edge_array_to_nef(neighbors, nef_indices)

        # Here, we compute the array that allows indexing into a flattened
        # version of the edge array (where the first two dimensions are merged):
        reverse_neighbor_index = (
            neighbors_index * neighbors_index.shape[1] + reversed_neighbor_list
        )
        # At this point, we have `reverse_neighbor_index[~nef_mask] = 0`, which however
        # creates too many of the same index which slows down backward enormously.
        # (See https://github.com/pytorch/pytorch/issues/41162)
        # We therefore replace the padded indices with a sequence of unique indices.
        reverse_neighbor_index[~nef_mask] = torch.arange(
            int(torch.sum(~nef_mask)), device=reverse_neighbor_index.device
        )


        use_manual_attention = edge_vectors.requires_grad and self.training

        node_features = self.node_embedder(atomic_numbers)
        if self.dens_enabled and "force_data" in data:
            noise_mask = (
                data["noise_mask"]
                if "noise_mask" in data
                else torch.zeros_like(atomic_numbers, dtype=torch.bool)
            )
            force_data = data["force_data"].to(node_features.dtype)
            force_norm = torch.linalg.vector_norm(
                force_data, dim=-1, keepdim=True
            )
            force_embedding = self.dens_force_encoder(
                torch.cat([force_data, force_norm], dim=-1)
            )
            node_features = node_features + force_embedding * noise_mask.unsqueeze(-1).to(
                node_features.dtype
            )

        cat = torch.cat([edge_vectors, edge_distances.unsqueeze(-1)], dim=-1) / (0.5 *self.cutoff)  # very rough normalization attempt
        edge_features = torch.concatenate([
            self.edge_center_embedder(element_indices_centers),
            self.edge_neighbor_embedder(element_indices_neighbors),
            self.edge_directional_embedder(cat)
        ], dim=-1)
        edge_features = self.edge_compressor(edge_features)
        # edge_features = edge_features + self.edge_mlp(edge_features)
        
        # print(input_node_embeddings.mean().item(), input_node_embeddings.std().item(), flush=True)
        # print("before", edge_features.mean().item(), (edge_features*cutoff_factors.unsqueeze(-1)).std().item(), flush=True)

        for combination_norm, combination_mlp, gnn_layer in zip(
            self.combination_norms, self.combination_mlps, self.gnn_layers, strict=True
        ):
            node_features, edge_features = gnn_layer(
                node_features,
                edge_features,
                element_indices_neighbors,
                edge_vectors,
                nef_mask,
                edge_distances,
                cutoff_factors,
                use_manual_attention,
            )

            # print("after gnn", edge_features.mean().item(), (edge_features*cutoff_factors.unsqueeze(-1)).std().item(), flush=True)

            # The GNN contraction happens by reordering the messages,
            # using a reversed neighbor list, so the new input message
            # from atom `j` to atom `i` in on the GNN layer N+1 is a
            # reversed message from atom `i` to atom `j` on the GNN layer N.
            corresponding_edge_features = edge_features.reshape(
                edge_features.shape[0] * edge_features.shape[1],
                edge_features.shape[2],
            )[reverse_neighbor_index].reshape(
                edge_features.shape[0],
                edge_features.shape[1],
                edge_features.shape[2],
            )
            concatenated = torch.cat(
                [edge_features, corresponding_edge_features], dim=-1
            )
            edge_features = edge_features + combination_mlp(combination_norm(concatenated))

            # print(input_node_embeddings.mean().item(), input_node_embeddings.std().item(), flush=True)
            # print("after combination", edge_features.mean().item(), (edge_features*cutoff_factors.unsqueeze(-1)).std().item(), flush=True)
        
        # print(flush=True)

        edge_features = edge_features * cutoff_factors.unsqueeze(-1)
        node_features = node_features + self.edge_expander(torch.sum(edge_features, dim=1))

        # TODO: this is messed up, you could do this for the energy
        structure_features = torch.index_add(
            torch.zeros((len(data["cell"]), self.node_to_edge_ratio * self.d_pet), device=node_features.device),
            0,
            data["batch"],
            node_features,
        )
        # divide by number of atoms
        atoms_bincount = torch.bincount(data["batch"], minlength=len(data["cell"]))
        structure_features = structure_features / atoms_bincount.unsqueeze(-1).clamp(min=1)

        outputs = {"node_features": node_features, "structure_features": structure_features}
        if displacement is not None:
            outputs["displacement"] = displacement
        return outputs

    def forward(self, data: AtomicData) -> dict[str, torch.Tensor]:
        if (self.regress_forces and not self.direct_forces) or (
            self.regress_stress and not self.direct_stress
        ):
            with torch.enable_grad():
                return self._forward_impl(data)
        return self._forward_impl(data)


@registry.register_model("PET_energy_head")
class PETEnergyHead(nn.Module, HeadInterface):
    # multi-layer perceptron on the node features
    def __init__(self, backbone: PETBackbone) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            Linear(backbone.node_to_edge_ratio * backbone.d_pet, backbone.feedforward_ratio * backbone.node_to_edge_ratio * backbone.d_pet),
            nn.SiLU(),
            Linear(backbone.feedforward_ratio * backbone.node_to_edge_ratio * backbone.d_pet, 1),
        )
    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        atomic_energies = self.mlp(emb["node_features"]).squeeze(-1)
        total_energies = torch.index_add(
            torch.zeros((len(data["cell"]),), device=atomic_energies.device),
            0,
            data["batch"],
            atomic_energies
        )
        return {"energy": total_energies}

@registry.register_model("PET_direct_force_head")
class PETDirectForceHead(nn.Module, HeadInterface):
    # multi-layer perceptron on the node features
    def __init__(self, backbone: PETBackbone) -> None:
        super().__init__()
        self.dens_enabled = backbone.dens_enabled
        self.mlp = nn.Sequential(
            Linear(backbone.node_to_edge_ratio * backbone.d_pet, backbone.feedforward_ratio * backbone.node_to_edge_ratio * backbone.d_pet),
            nn.SiLU(),
            Linear(backbone.feedforward_ratio * backbone.node_to_edge_ratio * backbone.d_pet, 3),
        )
        if self.dens_enabled:
            self.dens_mlp = nn.Sequential(
                Linear(backbone.node_to_edge_ratio * backbone.d_pet, backbone.feedforward_ratio * backbone.node_to_edge_ratio * backbone.d_pet),
                nn.SiLU(),
                Linear(backbone.feedforward_ratio * backbone.node_to_edge_ratio * backbone.d_pet, 3),
            )
    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        forces = self.mlp(emb["node_features"])
        if self.dens_enabled:
            noise_mask = (
                data["noise_mask"]
                if "noise_mask" in data
                else torch.zeros(
                    emb["node_features"].shape[0],
                    device=emb["node_features"].device,
                    dtype=torch.bool,
                )
            ).view(-1, 1)
            dens_forces = self.dens_mlp(emb["node_features"])
            forces = torch.where(noise_mask, dens_forces, forces)
        return {"forces": forces}
    

@registry.register_model("PET_direct_stress_head")
class PETDirectStressHead(nn.Module, HeadInterface):
    # multi-layer perceptron on the structure features
    def __init__(self, backbone: PETBackbone) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            Linear(backbone.node_to_edge_ratio * backbone.d_pet, backbone.feedforward_ratio * backbone.node_to_edge_ratio * backbone.d_pet),
            nn.SiLU(),
            Linear(backbone.feedforward_ratio * backbone.node_to_edge_ratio * backbone.d_pet, 9),
        )

    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        stress = self.mlp(emb["structure_features"])
        if "dens_batch_mask" in data:
            stress = torch.where(
                data["dens_batch_mask"].view(-1, 1),
                torch.zeros_like(stress),
                stress,
            )
        return {"stress": stress}


@registry.register_model("PET_grad_energy_force_stress_head")
class PETGradientEnergyForceStressHead(PETEnergyHead):
    def __init__(self, backbone: PETBackbone) -> None:
        super().__init__(backbone)
        self.regress_forces = backbone.regress_forces
        self.direct_forces = backbone.direct_forces
        self.regress_stress = backbone.regress_stress
        self.direct_stress = backbone.direct_stress
        self.dens_enabled = backbone.dens_enabled
        if self.dens_enabled:
            self.dens_mlp = nn.Sequential(
                Linear(
                    backbone.node_to_edge_ratio * backbone.d_pet,
                    backbone.feedforward_ratio * backbone.node_to_edge_ratio * backbone.d_pet,
                ),
                nn.SiLU(),
                Linear(backbone.feedforward_ratio * backbone.node_to_edge_ratio * backbone.d_pet, 3),
            )

    @conditional_grad(torch.enable_grad())
    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        atomic_energies = self.mlp(emb["node_features"]).squeeze(-1)
        energy = torch.index_add(
            torch.zeros((len(data["cell"]),), device=atomic_energies.device),
            0,
            data["batch"],
            atomic_energies,
        )
        outputs = {"energy": energy}

        if self.regress_forces and not self.direct_forces:
            if self.regress_stress and not self.direct_stress:
                grads = torch.autograd.grad(
                    [energy.sum()],
                    [data["pos"], emb["displacement"]],
                    create_graph=self.training,
                )
                forces = -grads[0]
                virial = grads[1].view(-1, 3, 3)
                volume = torch.det(data["cell"]).abs().unsqueeze(-1)
                stress = (virial / volume.view(-1, 1, 1)).view(-1, 9)
                if "dens_batch_mask" in data:
                    stress = torch.where(
                        data["dens_batch_mask"].view(-1, 1),
                        torch.zeros_like(stress),
                        stress,
                    )
                outputs["stress"] = stress
            else:
                forces = -torch.autograd.grad(
                    energy.sum(),
                    data["pos"],
                    create_graph=self.training,
                    retain_graph=True,
                )[0]

            if self.dens_enabled:
                noise_mask = (
                    data["noise_mask"]
                    if "noise_mask" in data
                    else torch.zeros(
                        emb["node_features"].shape[0],
                        device=emb["node_features"].device,
                        dtype=torch.bool,
                    )
                ).view(-1, 1)
                dens_forces = self.dens_mlp(emb["node_features"])
                forces = torch.where(noise_mask, dens_forces, forces)
            outputs["forces"] = forces

        elif self.regress_stress and not self.direct_stress:
            displacement = emb["displacement"]
            virial = torch.autograd.grad(
                energy.sum(),
                displacement,
                create_graph=self.training,
                retain_graph=True,
            )[0].view(-1, 3, 3)
            volume = torch.det(data["cell"]).abs().unsqueeze(-1)
            stress = (virial / volume.view(-1, 1, 1)).view(-1, 9)
            if "dens_batch_mask" in data:
                stress = torch.where(
                    data["dens_batch_mask"].view(-1, 1),
                    torch.zeros_like(stress),
                    stress,
                )
            outputs["stress"] = stress

        return {
            "energy": {"energy": outputs["energy"]},
            "forces": {"forces": outputs["forces"]},
            "stress": {"stress": outputs["stress"]},
        }
