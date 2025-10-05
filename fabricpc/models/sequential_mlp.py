import torch
import torch.nn as nn
from fabricpc.core.base_pc import PCNet
from fabricpc.core.sequential_pc import PCDenseLayer
from fabricpc.training import optimizers as pc_optim


# Define network structure
class PC_MLP(PCNet, nn.Module):
    """
    Initialize PC_MLP using a configuration dictionary.

    Expected keys in config:
      - dims_list: list[int], required
      - device: torch.device or str, optional (default: 'cuda' if available else 'cpu')
      - task_map: dict, required
      - Optional hyperparameters/flags (if present will be set on the instance):
          T_infer, eta_infer, eta_learn, latent_init_feedforward, etc.
    """

    def __init__(self, config: dict):
        super().__init__()
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")

        # Required parameters
        if "dims_list" not in config:
            raise ValueError("config['dims_list'] is required")
        if "task_map" not in config:
            raise ValueError("config['task_map'] is required")

        dims_list = config["dims_list"]
        task_map = config["task_map"]
        device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Core parameters
        self.dims_list = dims_list  # layer dimensions, bottom-up order
        self.module_list = []  # list of transforms between state layers
        # State variables
        self.z_latent = (
            []
        )  # list of tensors shape [batch_size, dim]. Each tensor is the state of a layer.
        self.error = []  # list of tensors shape [batch_size, dim]
        self.z_mu = []  # list of tensors shape [batch_size, dim]
        self.pre_activation_val = []  # list of tensors shape [batch_size, dim]
        self.gain_mod_error = []  # list of tensors shape [batch_size, dim]
        # Properties
        self.L = len(dims_list)  # number of state layers including input and output
        # Hyperparameters defaults
        self.T_infer = 20
        self.eta_infer = 0.1
        self.eta_learn = 1e-4
        # Process configuration
        self.task_map = task_map or {}  # dict {task_name: layer_index}
        self.clamp_dict = {}  # dict {layer_index: clamped_tensor}
        self.device = device
        self.optimizer = None  # {'type': 'adam', 'config': {'lr': self.eta_learn}}
        self.init_feedforward = (
            False  # Feedforward init of latents not implemented yet in PC_MLP
        )

        # Apply any extra config keys to existing attributes
        extra_cfg = {
            k: v
            for k, v in config.items()
            if k not in ("dims_list", "device", "task_map")
        }
        if extra_cfg:
            self.set_members(extra_cfg)

        # Build network transforms
        for layer_idx in range(0, self.L - 1):
            self.module_list.append(
                PCDenseLayer(
                    in_dim=dims_list[layer_idx + 1],
                    out_dim=dims_list[layer_idx],
                    device=self.device,
                )
            )

        # Attach optimizer
        for layer_idx in range(0, self.L - 1):
            self.module_list[layer_idx].optimizer = pc_optim.instantiate_optimizer(
                self.module_list[layer_idx].W, config.get("optimizer", None)
            )

    def allocate_layers(self, batch_size, device=None):
        if device is not None:
            self.device = device
        # Allocate tensors once; reuse and fill in-place during inference
        print(f"Allocating layers for batch size {batch_size} on device {self.device}")
        self.z_latent = [
            torch.empty(batch_size, dim, device=self.device) for dim in self.dims_list
        ]
        self.error = [
            torch.empty(batch_size, dim, device=self.device) for dim in self.dims_list
        ]
        self.z_mu = [
            torch.empty(batch_size, dim, device=self.device) for dim in self.dims_list
        ]
        self.pre_activation_val = [
            torch.empty(batch_size, dim, device=self.device) for dim in self.dims_list
        ]
        self.gain_mod_error = [
            torch.empty(batch_size, dim, device=self.device) for dim in self.dims_list
        ]

    def get_task_result(self, task_key):
        target_layer = self.task_map[task_key]
        if target_layer == 0:
            # Target state is a sink node; it has no latent state to be inferred. We get its predicted value by projection.
            return self.z_mu[0].clone()
        else:
            # Target state is a latent node, return its inferred value.
            return self.z_latent[target_layer].clone()

    def get_dim_for_key(self, key):
        # Sequential model key is an integer index of the layer
        if isinstance(key, int):
            return self.dims_list[key]
        raise ValueError(f"Unable to resolve dimension for latent layer {key}")

    def init_latents(self, clamp_dict, batch_size, device=None, std=0.05):
        # clamp_dict: dictionary of {layer_index: clamped_value}
        if device is not None:
            self.device = device
        self.clamp_dict = clamp_dict

        # Allocate tensors once; reuse and re-fill in-place on subsequent calls
        need_realloc = len(self.z_latent) != self.L or any(
            (t is None)
            or (not isinstance(t, torch.Tensor))
            or (t.device.type != self.device.type)
            or (t.size(0) != batch_size)
            for layer_idx, t in enumerate(self.z_latent)
        )
        if need_realloc:
            # Fresh allocation (first call or shape/device changed)
            self.allocate_layers(batch_size, device=self.device)
        # Initialize latents in-place
        for layer_idx in range(self.L):
            if layer_idx in self.clamp_dict:
                # Set clamped latent
                val = self.clamp_dict[layer_idx]
                if (
                    val.dim() != 2
                    or val.size(1) != self.dims_list[layer_idx]
                    or val.size(0) != batch_size
                ):
                    raise ValueError(
                        f"Clamp for layer {layer_idx} must have shape (B, {self.dims_list[layer_idx]}), got {tuple(val.shape)}"
                    )
                # In-place copy of clamped value
                self.z_latent[layer_idx].copy_(val)
            else:
                # Reinitialize latent with random values in-place
                self.z_latent[layer_idx].normal_(mean=0.0, std=std)

    def update_projections(self):
        """
        Project all layers to their predicted values
        Computes:
            pre_activation_val = z_{l+1} * W_{l}
            z_mu = f( pre_activation_val )
        """
        for layer_idx in range(self.L):
            # Handle source nodes (no incoming connections)
            if layer_idx == (self.L - 1):
                # This is the top level latent. There are no predictors above; set z_mu to zeros.
                self.pre_activation_val[layer_idx].zero_()
                self.z_mu[layer_idx].zero_()
            else:
                # Predict z_{l} from z_{l+1}
                module = self.module_list[layer_idx]
                # Call to module returns:
                #   a = z_{l+1} * W_{l}     preactivations a_{l}
                #   z_mu = f_{l}(a)        predictions \hat z_{l}
                self.z_mu[layer_idx], self.pre_activation_val[layer_idx] = (
                    module.forward(self.z_latent[layer_idx + 1])
                )  # from latent of higher level, predict the lower level

    def update_error(self):
        """
        Compute prediction errors and gain-modulated errors for all layers
        Computes:
            error: list of prediction errors for each layer
            gain_mod_error: list of gain-modulated errors for each layer
        """
        for layer_idx in range(self.L):
            if layer_idx == (self.L - 1):
                # This is the top level latent. No incoming connections; define zero error.
                self.error[layer_idx].zero_()
                self.gain_mod_error[layer_idx].zero_()
                continue
            if layer_idx == 0 and (0 not in self.clamp_dict):
                # The sink node is floating, the error is zero
                self.error[layer_idx].zero_()
                self.gain_mod_error[layer_idx].zero_()
                continue
            # General case: layers have incoming connections and thus prediction errors
            act_derivative = self.module_list[layer_idx].activation_deriv(
                self.pre_activation_val[layer_idx]
            )  # Activation derivative
            self.error[layer_idx] = self.z_latent[layer_idx] - self.z_mu[layer_idx]
            self.gain_mod_error[layer_idx] = self.error[layer_idx] * act_derivative

    def update_latents_step(self):
        """
        Update latent states for all nodes; one inference step
        gradient dE/dz_{l} = eps_{l} - [eps_{l-1} .* f'_{l-1}(z_{l} W_{l-1})] * [W_{l-1}^T]
        """
        for layer_idx in range(self.L):
            if layer_idx in self.clamp_dict:
                continue  # Don't update latent of a clamped layer
            if layer_idx == 0:
                # Bottom layer has no outgoing links to below. Its gradient is just the local prediction error.
                dE_dz_l = self.error[layer_idx]
            else:
                # Layers greater than zero have contributions from local error and the projection to the layer below.
                dE_dz_l = self.error[layer_idx] - torch.matmul(
                    self.gain_mod_error[layer_idx - 1],
                    self.module_list[layer_idx - 1].W.T,
                )
            # Update latent z_{l}
            self.z_latent[layer_idx] -= self.eta_infer * dE_dz_l

    def update_weights(self):
        """
        Update weights for all transfer functions between layers
        gradient dE/dW_{l} = -[z_{l+1}^T] * [eps_{l} .* f'_{l+1}(z_{l+1} * W_{l})]
        """
        for layer_idx in range(self.L - 1):
            dE_dW_l = -torch.matmul(
                self.z_latent[layer_idx + 1].T, self.gain_mod_error[layer_idx]
            )
            # self.module_list[layer_idx].W -= self.eta_learn * dE_dW_l
            self.module_list[layer_idx].optimizer.step(dE_dW_l)

    def infer(
        self, clamps_dict, energy_record: list = None, selection_list: list = None
    ):
        self.clamp_dict = clamps_dict
        for t in range(self.T_infer):
            self.update_projections()  # Get projected states (z_mu)
            self.update_error()  # Measure error
            self.get_total_energy(energy_record, selection_list)
            self.update_latents_step()  # Inference step t

    def learn(self, energy_record: list = None, selection_list: list = None):
        # Learning loop for weights (single step)
        self.update_projections()  # Get projected states (z_mu)
        self.update_error()  # Measure error
        self.get_total_energy(energy_record, selection_list)
        self.update_weights()  # Learning weights step        with torch.no_grad():

    def get_total_energy(self, energy_record: list, selection_list: list):
        if energy_record is not None:
            if selection_list is None:
                selection_list = [i for i in range(self.L)]
            B = self.error[0].size(0)  # Batch size for normalization
            latent_energy = (
                0.5
                * sum(
                    err.pow(2).sum().item()
                    for i, err in enumerate(self.error)
                    if i in selection_list
                )
                / B
            )
            energy_record.append(latent_energy)
