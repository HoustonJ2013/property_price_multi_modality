from torch.nn import Linear, Module, ModuleList, LayerNorm, GELU, Sequential, Parameter, TransformerEncoderLayer
from typing import Any, Dict, List
from torch import Tensor
import torch 

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.conv import TabTransformerConv, FTTransformerConvs
from torch_frame.nn.conv import TableConv
from torch_frame.nn.conv.tab_transformer_conv import FFN, SelfAttention
from torch_frame.nn.encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    LinearEmbeddingEncoder,
    StypeWiseFeatureEncoder,
    MultiCategoricalEmbeddingEncoder,
    TimestampEncoder,
)


stype_encoder_dict_1 = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: LinearEncoder(),
    stype.timestamp: TimestampEncoder()
}

stype_encoder_dict_2 = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: LinearEncoder(),
    stype.multicategorical: MultiCategoricalEmbeddingEncoder(),
    stype.timestamp: TimestampEncoder()
}


stype_encoder_dict_3 = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: LinearEncoder(),
    stype.multicategorical: MultiCategoricalEmbeddingEncoder(),
    stype.embedding: LinearEmbeddingEncoder(),
    stype.timestamp: TimestampEncoder()
}


class TabTransformer(Module):
    def __init__(
        self,
        channels: int,
        num_layers: int,
        num_heads: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
        stype_encoder_dict, 
        out_channels: int = 1,
        user_conv_v2: bool = False,
    ):
        super().__init__()
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        if user_conv_v2:
            self.tab_transformer_convs = ModuleList([
                TabTransformerConv_v2(
                    channels=channels,
                    num_heads=num_heads,
                ) for _ in range(num_layers)
            ])
        else:
            self.tab_transformer_convs = ModuleList([
                TabTransformerConv(
                    channels=channels,
                    num_heads=num_heads,
                ) for _ in range(num_layers)
            ])
        self.decoder = Linear(channels, out_channels)

    def forward(self, tf: TensorFrame) -> Tensor:
        x, _ = self.encoder(tf)
        for tab_transformer_conv in self.tab_transformer_convs:
            x = tab_transformer_conv(x)
        out = self.decoder(x.mean(dim=1)).squeeze()
        return out
    

class TabTransformerConv_v2(TableConv):
    r"""The TabTransformer Layer introduced in the
    `"TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
    <https://arxiv.org/abs/2012.06678>`_ paper.

    Args:
        channels (int): Input/output channel dimensionality
        num_heads (int): Number of attention heads
        attn_dropout (float): attention module dropout (default: :obj:`0.`)
        ffn_dropout (float): attention module dropout (default: :obj:`0.`)
    """
    def __init__(self, channels: int, num_heads: int, attn_dropout: float = 0.,
                 ffn_dropout: float = 0.):
        super().__init__()
        self.norm_1 = LayerNorm(channels)
        self.attn = SelfAttention(channels, num_heads, attn_dropout)
        self.norm_2 = LayerNorm(channels)
        self.ffn = FFN(channels, dropout=ffn_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm_1(x)
        x = x +self.attn(x)
        x = self.norm_2(x)
        x = x + self.ffn(x)
        return x

    def reset_parameters(self):
        self.norm_1.reset_parameters()
        self.attn.reset_parameters()
        self.norm_2.reset_parameters()
        self.ffn.reset_parameters()


class TabTransformer_v2(Module):
    def __init__(
        self,
        channels: int,
        num_layers: int,
        num_heads: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
        stype_encoder_dict, 
        out_channels: int = 1,
    ):
        super().__init__()
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        self.tab_transformer_convs = ModuleList([
            TabTransformerConv_v2(
                channels=channels,
                num_heads=num_heads,
            ) for _ in range(num_layers)
        ])
        self.cls_embedding = Parameter(torch.empty(channels))
        self.decoder = Linear(channels, out_channels)


    def forward(self, tf: TensorFrame) -> Tensor:
        x, _ = self.encoder(tf)

        B, _, _ = x.shape
        x_cls = self.cls_embedding.repeat(B, 1, 1)
        x_concat = torch.cat([x_cls, x], dim=1)

        for tab_transformer_conv in self.tab_transformer_convs:
            x_concat = tab_transformer_conv(x_concat)
        x_cls = x_concat[:, 0, :]
        out = self.decoder(x_cls).squeeze()
        return out



class FTTransformer(Module):
    def __init__(
        self,
        channels: int,
        num_layers: int,
        num_heads: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
        stype_encoder_dict,
        out_channels: int = 1,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        self.ftt_transformer_convs = ModuleList([
                TransformerEncoderLayer(
                    d_model=channels,
                    nhead=num_heads,
                    dim_feedforward=channels*mlp_ratio, 
                    activation="gelu", 
                    batch_first=True
                ) for _ in range(num_layers)
            ])
        self.cls_embedding = Parameter(torch.empty(channels))
        self.decoder = Sequential(
                        LayerNorm(channels),
                        GELU(),
                        Linear(channels, channels*mlp_ratio),
                        GELU(),
                        Linear(channels*mlp_ratio, out_channels),
        )

    def forward(self, tf: TensorFrame) -> Tensor:
        x, _ = self.encoder(tf)

        B, _, _ = x.shape
        x_cls = self.cls_embedding.repeat(B, 1, 1)
        x_concat = torch.cat([x_cls, x], dim=1)

        for ftt_transformer_conv in self.ftt_transformer_convs:
            x_concat = ftt_transformer_conv(x_concat)
        x_cls = x_concat[:, 0, :]
        out = self.decoder(x_cls).squeeze()
        return out