from torch.nn import Linear, Module, ModuleList
from typing import Any, Dict, List
from torch import Tensor
import torch 

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.conv import TabTransformerConv, FTTransformerConvs
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
    ):
        super().__init__()
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
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
    

class FTTransformer(Module):
    def __init__(
        self,
        channels: int,
        num_layers: int,
        num_heads: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[torch_frame.stype, List[str]],
        out_channels: int = 1,
    ):
        super().__init__()
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict={
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(), 
                stype.multicategorical: MultiCategoricalEmbeddingEncoder(),
            },
        )
        self.ftt_transformer_conv = FTTransformerConvs(channels=channels,
                                                       num_layers=num_layers,
                                                       nhead=num_heads)

        self.decoder = Linear(channels, out_channels)

    def forward(self, tf: TensorFrame) -> Tensor:
        x, _ = self.encoder(tf)
        x, x_cls = self.ftt_transformer_conv(x)
        x_cls = x_cls.unsqueeze(1)
        x_concat = torch.cat([x_cls, x], dim=1)
        out = self.decoder(x_concat.mean(dim=1)).squeeze()
        return out