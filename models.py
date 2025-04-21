import math
from typing import Optional, List, Union, Tuple
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from tcn import TCN, tcn_full_summary

from torch_geometric.data import Data
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.transforms import LaplacianLambdaMax
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, get_laplacian, softmax

class GATv2Conv(MessagePassing):
    r"""Implementing the GATv2 operator from the `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper.
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class ASTGCNBlock(nn.Module):
    r"""An implementation of the Attention Based Spatial-Temporal Graph Convolutional Block.
    For details see this paper: `"Attention Based Spatial-Temporal Graph Convolutional
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_
    """

    def __init__(
        self,
        in_channels: int,
        K: int,
        nb_chev_filter: int,
        nb_time_filter: int,
        time_strides: int,
        num_of_vertices: int,
        num_of_timesteps: int,
        nb_gatv2conv: int,
        dropout_gatv2conv: float = 0.5,
        head_gatv2conv: int = 4,
        normalization: Optional[str] = None,
        bias: bool = True,
    ):
        super(ASTGCNBlock, self).__init__()

        self._gatv2conv_attention = GATv2Conv(
            in_channels, out_channels=nb_gatv2conv, dropout=dropout_gatv2conv, heads=head_gatv2conv
        )
        self._time_convolution = nn.Conv2d(
            nb_gatv2conv * head_gatv2conv,
            nb_time_filter,
            kernel_size=(1, 3),
            stride=(1, time_strides),
            padding=(0, 1),
        )
        self._residual_convolution = nn.Conv2d(
            in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides)
        )
        self._layer_norm = nn.LayerNorm(nb_time_filter)
        self._normalization = normalization

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: Union[torch.LongTensor, List[torch.LongTensor]],
    ) -> torch.FloatTensor:
        """
        Making a forward pass with the ASTGCN block.
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = X.shape # (32, 307, 1, 12)

        if not isinstance(edge_index, list):
            data = Data(
                edge_index=edge_index, edge_attr=None, num_nodes=num_of_vertices
            )

            X_hat = []
            for t in range(num_of_timesteps):
                X_hat.append(
                    torch.unsqueeze(
                        self._gatv2conv_attention(x=X[0, :, :, t], edge_index=edge_index),
                        -1,
                    )
                )

            X_hat = F.relu(torch.cat(X_hat, dim=-1))
        else:
            X_hat = []
            for t in range(num_of_timesteps):
                data = Data(
                    edge_index=edge_index[t], edge_attr=None, num_nodes=num_of_vertices
                )
                X_hat.append(
                    torch.unsqueeze(
                        self._gatv2conv_attention(x=X[0, :, :, t], edge_index=edge_index[t]),
                        -1,
                    )
                )
            X_hat = F.relu(torch.cat(X_hat, dim=-1))

        X_hat = X_hat[None, ...]
        X_hat = self._time_convolution(X_hat.permute(0, 2, 1, 3)) 

        X = self._residual_convolution(X.permute(0, 2, 1, 3))  

        X = self._layer_norm(F.relu(X + X_hat).permute(0, 3, 2, 1))
        X = X.permute(0, 2, 3, 1)
        return X


class ASTGCN(nn.Module):
    r"""An implementation of the Attention Based Spatial-Temporal Graph Convolutional Cell.
    For details see this paper: `"Attention Based Spatial-Temporal Graph Convolutional
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_
    """

    def __init__(
        self,
        nb_block: int,
        in_channels: int,
        K: int,
        nb_chev_filter: int,
        nb_time_filter: int,
        time_strides: int,
        num_for_predict: int,
        len_input: int,
        num_of_vertices: int,
        nb_gatv2conv: int,
        dropout_gatv2conv: float = 0.5,
        head_gatv2conv: int = 4,
        normalization: Optional[str] = None,
        bias: bool = True,
    ):

        super(ASTGCN, self).__init__()

        self._blocklist = nn.ModuleList(
            [
                ASTGCNBlock(
                    in_channels,
                    K,
                    nb_chev_filter,
                    nb_time_filter,
                    time_strides,
                    num_of_vertices,
                    len_input,
                    nb_gatv2conv,
                    dropout_gatv2conv,
                    head_gatv2conv,
                    normalization,
                    bias,
                )
            ]
        )

        self._blocklist.extend(
            [
                ASTGCNBlock(
                    nb_time_filter,
                    K,
                    nb_chev_filter,
                    nb_time_filter,
                    1,
                    num_of_vertices,
                    len_input // time_strides,
                    nb_gatv2conv,
                    dropout_gatv2conv,
                    head_gatv2conv,
                    normalization,
                    bias,
                )
                for _ in range(nb_block - 1)
            ]
        )

        self._final_conv = nn.Conv2d(
            int(len_input / time_strides),
            num_for_predict,
            kernel_size=(1, nb_time_filter),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Resetting the parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self, X: torch.FloatTensor, edge_index: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        Making a forward pass.

        """
        for block in self._blocklist:
            X = block(X, edge_index) 

        X = self._final_conv(X.permute(0, 3, 1, 2))

        X = X[:, :, :, -1]
        X = X.permute(0, 2, 1)
        return X

class GATv2TCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        len_input: int,
        len_output: int,
        temporal_filter: int,
        out_gatv2conv: int,
        dropout_tcn: float = 0.5,
        dropout_gatv2conv: float = 0.5,
        head_gatv2conv: int = 1
    ):
        super(GATv2TCN, self).__init__()
        self._gatv2conv_attention = GATv2Conv(
            in_channels, out_channels=out_gatv2conv, dropout=dropout_gatv2conv, heads=head_gatv2conv
        )

        self._time_convolution = nn.Conv2d(
            out_gatv2conv*head_gatv2conv,
            temporal_filter,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        self._residual_convolution = nn.Conv2d(
            in_channels, temporal_filter, kernel_size=(1, 1), stride=(1, 1)
        )
        self._layer_norm = nn.LayerNorm(temporal_filter)
        self._final_conv = nn.Conv2d(
            len_input,
            out_channels,
            kernel_size=(1, temporal_filter//len_output),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: Union[torch.LongTensor, List[torch.LongTensor]],
    ) -> torch.FloatTensor:
        assert isinstance(edge_index, list)
        X_hat = []
        for t in range(len(edge_index)):
            X_hat.append(
                torch.unsqueeze(
                    self._gatv2conv_attention(x=X[0, :, :, t], edge_index=edge_index[t]),
                    -1,
                )
            )

        X_hat = F.relu(torch.cat(X_hat, dim=-1))[None, ...]
        X_hat = self._time_convolution(X_hat.permute(0, 2, 1, 3))
        X = self._residual_convolution(X.permute(0, 2, 1, 3))

        X = self._layer_norm(F.relu(X + X_hat).permute(0, 3, 2, 1))

        X = self._final_conv(X)
        return X.permute(0, 2, 1, 3)[..., -1]
