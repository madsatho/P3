import torch
from torch import nn #<--- Fordi jeg ik vil skrive torch.nn.Module hver gang
from torch.nn import AdaptiveAvgPool2d


#Model struktur inspireret fra https://www.learnpytorch.io/ af forfatter Daniel Bork

class GraveXModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            #nn.ReLU(), # For at gør lortet ikke-lineær
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)



class GraveXModelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 64 * 64, out_features=output_shape),
            #nn.Linear(in_features=hidden_units, out_features=output_shape)
        )


    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

def conv_layer(input_shape: int, output_shape: int):
    return nn.Sequential(
        nn.Conv2d(input_shape, output_shape, 3, padding=1),
        nn.BatchNorm2d(output_shape),
        nn.ReLU(),
        nn.Conv2d(output_shape, output_shape, 3, padding=1),
        nn.BatchNorm2d(output_shape),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class GraveXModelV3(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            conv_layer(input_shape, hidden_units),
            conv_layer(hidden_units, hidden_units*2),
            conv_layer(hidden_units*2, hidden_units*4),
            conv_layer(hidden_units*4, hidden_units*8),   #Conv2d tager KANALER in, så vi fordobler antallet af kanaler hver gang!
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(2048 , output_shape)
        )

    def forward(self, x):
        return self.classifier(self.layer_stack(x))
