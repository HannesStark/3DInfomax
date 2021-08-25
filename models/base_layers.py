import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'SiLU', 'None'}
EPS = 1e-5


def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str), 'Unhandled activation function'
    activation = activation[0]
    if activation.lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation]()


class FCLayer(nn.Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:
    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)
    Arguments
    ----------
        in_dim: int
            Input dimension of the layer (the torch.nn.Linear)
        out_dim: int
            Output dimension of the layer.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        batch_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_dim}}`
            (Default value = None)
    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        batch_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_dim: int
            Input dimension of the linear layer
        out_dim: int
            Output dimension of the linear layer
    """

    def __init__(self, in_dim, out_dim, activation='relu', dropout=0., batch_norm=False, batch_norm_momentum=0.1,
                 bias=True, init_fn=None,
                 device='cpu'):
        super(FCLayer, self).__init__()
        self.__params = locals()
        del self.__params['__class__']
        del self.__params['self']
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.linear = nn.Linear(in_dim, out_dim, bias=bias).to(device)
        self.dropout = None
        self.batch_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_dim, momentum=batch_norm_momentum).to(device)
        self.activation = get_activation(activation)
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_dim)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.batch_norm is not None:
            if h.shape[1] != self.out_dim:
                h = self.batch_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.batch_norm(h)
        return h


class MLP(nn.Module):
    """
        Simple multi-layer perceptron, built of a series of FCLayers
    """

    def __init__(self, in_dim, out_dim, layers, hidden_size=None, mid_activation='relu', last_activation='none',
                 dropout=0., mid_batch_norm=False, last_batch_norm=False, batch_norm_momentum=0.1, device='cpu'):
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.out_dim = out_dim

        self.fully_connected = nn.ModuleList()
        if layers <= 1:
            self.fully_connected.append(FCLayer(in_dim, out_dim, activation=last_activation, batch_norm=last_batch_norm,
                                                device=device, dropout=dropout,
                                                batch_norm_momentum=batch_norm_momentum))
        else:
            self.fully_connected.append(
                FCLayer(in_dim, hidden_size, activation=mid_activation, batch_norm=mid_batch_norm,
                        device=device, dropout=dropout, batch_norm_momentum=batch_norm_momentum))
            for _ in range(layers - 2):
                self.fully_connected.append(FCLayer(hidden_size, hidden_size, activation=mid_activation,
                                                    batch_norm=mid_batch_norm, device=device, dropout=dropout,
                                                    batch_norm_momentum=batch_norm_momentum))
            self.fully_connected.append(
                FCLayer(hidden_size, out_dim, activation=last_activation, batch_norm=last_batch_norm,
                        device=device, dropout=dropout, batch_norm_momentum=batch_norm_momentum))

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y