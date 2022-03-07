import torch
import copy

class ResBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same')
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same') # in the future replace with dilated conv?
        self.bn1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.downsample = torch.nn.MaxPool2d(kernel_size=4)
        self.activation = torch.nn.PReLU()

    def forward(self, x):
        input = copy.copy(x)  # backup the input for later
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.cat((x, input), dim=1)
        x = self.downsample(x)
        x = self.activation(x)
        return x





class Rcnn(torch.nn.Module):
    def __init__(self,channels=[64,32,16],features=[]):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """

        super().__init__()
        self.channels=channels
        depth=len(channels)
        self.conv=torch.nn.ModuleList([ResBlock(3,channels[1])]+[ResBlock(sum(channels[0:i+1]),channels[i+1]) for i in range(1,depth-1)])

        self.linear=torch.nn.ModuleList([torch.nn.Linear(features[i],features[i+1]) for i in range(0,len(features)-1)])
    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 4, 5
        and reuse the e parameter to compute the contribution of these orders.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same parameter many
        times when defining a computational graph.
        """

        for conv in self.conv :
            x=conv(x)
        x=x.flatten()
        for linear in self.linear :
            x=linear(x)

        return x

