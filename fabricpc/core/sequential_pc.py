import torch
import torch.nn as nn


## SEQUENTIAL LAYERED PCN
# Define latent layer
class PCDenseLayer(nn.Module):
    def __init__(
        self,
        in_dim,  # d_{l+1}  - dimension of layer above
        out_dim,  # d_l      - dimension of current layer
        activation_fn=torch.relu,  # nonlinearity f^(l)
        activation_deriv=lambda a: (a > 0).float(),  # derivative f^(l)'
        device=torch.device("cuda"),
        init_std=0.05,
    ):
        super().__init__()
        # Initialize weights
        self.W = init_std * torch.randn(in_dim, out_dim, device=device)

        self.activation_fn = activation_fn
        self.activation_deriv = activation_deriv
        self.device = device
        self.optimizer = None  # instance of an optimizer

    def forward(self, x_above):
        # Projection to the layer below
        # x_{l+1}: [batch_size, d_{l+1}]
        # W: [d_{l+1}, d_{l}]
        a = torch.matmul(x_above, self.W)  # a_{l} = X_{l+1} W_{l}
        z_mu = self.activation_fn(a)  # \hat X_{l} = f_{l}( a_{l} )
        # return: [batch_size, d_{l}]
        return z_mu, a  # also return the preactivations for computing the gradient
