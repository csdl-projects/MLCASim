##################################################################
## Base layer code to consist CircuitMixer                      ##
## CircuitMixer is a Mixer based prediction model               ##
## CircuitNorm2d : Custom normalization layer                   ##
## MixerLayer : Mixer layer for CircuitMixer                    ##
## ConditionalMixerLayer : Mixer layer with conditional input   ##
## DataMixing : Data mixing layer (time)                        ##
## FeatureMixing : Feature mixing layer (channel)               ##
##################################################################
## Author: Jaeseung Lee                                         ##
## Date: July 2024                                              ##
## Affiliation: POSTECH CSDL, South Korea                       ##
##################################################################

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

class CircuitNorm2d(nn.BatchNorm1d):
    def __init__(self, normalized_shape: tuple[int, int]):
        num_data, num_channels = normalized_shape
        super().__init__(num_data * num_channels)
        self.num_data = num_data
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], -1, 1)
        x = super().forward(x)
        x = x.reshape(x.shape[0], self.num_data, self.num_channels)
        return x
    
class MixerLayer(nn.Module):
    def __init__(
        self,
        data_length: int,
        input_channel: int,
        output_channel: int,
        hidden_size: int,
        activation: Callable[[Tensor], Tensor] = F.relu,
        dropout_rate: float = 0.1,
        is_norm_before: bool = False,
        norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.data_mixing = DataMixing(
            data_length,
            input_channel,
            activation,
            dropout_rate,
            norm_type = norm_type,
        )
        self.param_mixing = FeatureMixing(
            data_length,
            input_channel,
            output_channel,
            hidden_size,
            activation,
            dropout_rate,
            norm_type = norm_type,
            is_norm_before= is_norm_before
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.data_mixing(x)
        x = self.param_mixing(x)
        return x

class ConditionalMixerLayer(nn.Module):
    def __init__(
        self,
        data_length: int,
        input_channel: int,
        output_channel: int,
        param_length: int,
        hidden_size: int,
        activation: Callable[[Tensor], Tensor] = F.relu,
        dropout_rate: float = 0.1,
        is_norm_before: bool = False,
        norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.data_mixing = DataMixing(
            data_length,
            input_channel,
            activation,
            dropout_rate,
            norm_type = norm_type,
        )
        self.param_mixing = ConditionalFeatureMixing(
            data_length,
            input_channel,
            output_channel=output_channel,
            param_length=param_length,
            hidden_size=hidden_size,
            activation=activation,
            dropout_rate=dropout_rate,
            is_norm_before= is_norm_before,
            norm_type = norm_type,
        )

    def forward(self, x: Tensor, x_param: Tensor) -> Tensor:
        x = self.data_mixing(x)
        x, _ = self.param_mixing(x, x_param)
        return x

class DataMixing(nn.Module):
    def __init__(
        self,
        data_length: int,
        input_channel: int,
        activation: Callable[[Tensor], Tensor] = F.relu,
        dropout_rate: float = 0.1,
        norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()

        self.fc1 = nn.Linear(data_length, data_length)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = norm_type((data_length, input_channel))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = feature_to_data(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = data_to_feature(x)
        return self.norm(residual + x)

class FeatureMixing(nn.Module):
    def __init__(
        self,
        data_length: int,
        input_channel: int,
        output_channel: int,
        hidden_size: int,
        activation: Callable[[Tensor], Tensor] = F.relu,
        dropout_rate: float = 0.1,
        is_norm_before: bool = True,
        norm_type: type[nn.Module] = CircuitNorm2d,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_channel, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_channel)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = (
            nn.Linear(input_channel, output_channel)
            if input_channel != output_channel
            else nn.Identity()
        )
        self.norm_before = (
            norm_type((data_length, input_channel))
            if is_norm_before
            else nn.Identity()
        )
        self.norm_after = (
            norm_type((data_length, output_channel))
            if not is_norm_before
            else nn.Identity()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        residual = self.projection(x)
        x = self.norm_before(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return self.norm_after(x + residual)

class ConditionalFeatureMixing(nn.Module):
    def __init__(
        self,
        data_length: int,
        input_channel: int,
        output_channel: int,
        param_length: int,
        hidden_size: int,
        activation: Callable[[Tensor], Tensor] = F.relu,
        dropout_rate: float = 0.1,
        is_norm_before: bool = False,
        norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.fc = nn.Linear(param_length, output_channel)
        self.feature_mixing = FeatureMixing(
            data_length = data_length,
            input_channel = input_channel + output_channel,
            output_channel = output_channel,
            hidden_size = hidden_size,
            activation = activation,
            dropout_rate = dropout_rate,
            is_norm_before = is_norm_before,
            norm_type = norm_type,
        )

    def forward(self, x: Tensor, param: Tensor) -> Tensor:
        f = self.fc(param)
        f = f.unsqueeze(1).repeat(
            1, x.shape[1], 1
        )
        return (
            self.feature_mixing(
            torch.cat([x, f], dim = -1),
        ), f.detach())

def data_to_feature(x: Tensor) -> Tensor:
    return x.permute(0, 2, 1)

def feature_to_data(x: Tensor) -> Tensor:
    return x.permute(0, 2, 1)