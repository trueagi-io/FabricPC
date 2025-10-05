from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from fabricpc.core.activation_functions import get_activation
from fabricpc.training import optimizers as pc_optim


class GraphElement(ABC):
    def __init__(self):
        self.name = ""  # unique name of the element

    def set_members(self, dict):
        for key, value in dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Attribute {key} not found in class {self.__class__.__name__}"
                )


@dataclass(frozen=True)
class EdgeId:
    source: str
    target: str
    slot: str

    def key(self) -> str:
        return f"{self.source}->{self.target}:{self.slot}"


class PCNodeBase(GraphElement):
    def __init__(self, config: dict):
        super().__init__()
        if "name" not in config:
            raise ValueError("config['name'] is required")
        if "dim" not in config:
            raise ValueError("config['dim'] is required")
        if "activation" not in config:
            raise ValueError("config['activation'] is required")

        # Core parameters
        self.dim = 0  # length of the state vector
        self.activation_fn = None  # activation function
        self.activation_deriv = None  # activation derivative
        self.slots = {}  # dictionary of {slot_name: slot_object}
        # State variables
        self.z_latent = None  # tensor shape [batch_size, dim]
        self.error = None  # tensor shape [batch_size, dim]
        self.z_mu = None  # tensor shape [batch_size, dim]
        self.pre_activation_val = None  # tensor shape [batch_size, dim]
        self.gain_mod_error = None  # tensor shape [batch_size, dim]
        # Properties
        self.in_neighbors = (
            {}
        )  # dictionary of incoming edges {EdgeId obj: in-neighbor node obj}
        self.out_neighbors = (
            {}
        )  # dictionary of outgoing edges {EdgeId obj: out-neighbor node obj}
        self.out_degree = 0  # number of outgoing edges
        self.in_degree = 0  # number of incoming edges
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = torch.float32
        self.optimizer = None

    def set_activation_from_config(self, config: dict):
        self.activation_fn, self.activation_deriv = get_activation(config)

    def register_in_neighbor(self, edge: EdgeId, source_node: PCNodeBase):
        if edge.target != self.name:
            raise ValueError(f"Edge {edge.key()} is not directed to node {self.name}")
        if edge.slot not in self.slots:
            raise ValueError(
                f"Slot {edge.slot} not found in node {self.name}. Expected one of: {self.slots.keys()}"
            )

        idx_start = sum([node.dim for node in self.in_neighbors.values()])
        idx_end = idx_start + source_node.dim
        self.weight_map[edge] = [idx_start, idx_end]
        self.in_neighbors[edge] = source_node
        self.slots[edge.slot].add_connection(edge, source_node)
        self.in_degree += 1

    def register_out_neighbor(self, edge: EdgeId, target_node: PCNodeBase):
        if edge.source != self.name:
            raise ValueError(f"Edge {edge.key()} is not directed from node {self.name}")
        self.out_neighbors[edge] = target_node
        self.out_degree += 1

    def allocate_state(self, batch_size: int):
        shape = (batch_size, self.dim)
        self.z_latent = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.error = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.z_mu = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.pre_activation_val = torch.zeros(
            shape, device=self.device, dtype=self.dtype
        )
        self.gain_mod_error = torch.zeros(shape, device=self.device, dtype=self.dtype)

    @abstractmethod
    def define_slots(self):
        pass

    @abstractmethod
    def compute_projection(self):
        pass

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def setup_optimizer(self, optim_config: dict):
        pass

    @abstractmethod
    def update_weights(self):
        pass

    @abstractmethod
    def get_jacobian(self, edge: EdgeId):
        pass


class InputSlot(GraphElement):
    """
    Abstract base class for input slots of a PCNodeBase.
    """

    def __init__(self):
        super().__init__()
        self.name = ""
        self.parent_node = None
        self.in_neighbors = (
            {}
        )  # dictionary of incoming edges {EdgeId obj: in-neighbor node obj}
        self.is_multi_input = None  # bool: allow multiple inputs to a slot

    def is_multi_input(self):
        return self.is_multi_input

    @abstractmethod
    def add_connection(self, edge: EdgeId, source_node: PCNodeBase):
        pass


class SingleIn(InputSlot):
    def __init__(self, slot_name: str, node_obj: PCNodeBase):
        super().__init__()
        self.name = slot_name
        self.parent_node = node_obj
        self.is_multi_input = False

    def add_connection(self, edge: EdgeId, source_node: PCNodeBase):
        """Allow only a single edge to connect to this slot"""
        if len(self.in_neighbors) > 0:
            raise ValueError(
                f"Single-input slot {self.name} in parent node {self.parent_node.name} already has a connection from another node {next(iter(self.in_neighbors.values())).name}"
            )
        self.in_neighbors[edge] = source_node


class MultiIn(InputSlot):
    def __init__(self, slot_name: str, node_obj: PCNodeBase):
        super().__init__()
        self.name = slot_name
        self.parent_node = node_obj
        self.is_multi_input = True

    def add_connection(self, edge: EdgeId, source_node: PCNodeBase):
        """Append any non-duplicated edge to this slot."""
        if edge in self.in_neighbors:
            raise ValueError(
                f"Slot {self.name} in parent node {self.parent_node.name} already has a connection from node {edge.source}"
            )
        self.in_neighbors[edge] = source_node


class LinearPCNode(PCNodeBase):
    def __init__(self, config: dict):
        super().__init__(config)
        # Transfer function parameters
        self.W = (
            None  # concatenated weights matrix, shape [sum of source.dim, self.dim]
        )
        self.b = None  # bias term, shape [1, dim_target]
        self.use_bias = True  # whether to use bias term in projections
        self.weight_map = (
            {}
        )  # dictionary of {EdgeId obj: [idx_start, idx_end] indices into concatenated weights matrix}
        # State variables
        self.inputs_all_concat = None  # tensor shape [batch_size, sum of source.dim]
        # Optim
        self.optimizer = None  # weights optimizer
        self.bias_optimizer = None
        # Hyperparameters
        self.weight_init_std = 0.05
        self.weight_init_method = "normal"

        # Construct objects from config
        self.set_activation_from_config(config["activation"])
        config = {k: v for k, v in config.items() if k != "activation"}

        # Finally, apply any extra config keys to existing attributes
        self.set_members(config)
        self.define_slots()

    def define_slots(self):
        slot_list = [MultiIn(slot_name="in", node_obj=self)]  # one multi-input slot
        self.slots = {slot.name: slot for slot in slot_list}

    def allocate_state(self, batch_size: int):
        super().allocate_state(batch_size)
        n_features = sum([node.dim for node in self.in_neighbors.values()])
        self.inputs_all_concat = torch.zeros(
            [batch_size, n_features], device=self.device
        )

    def compute_projection(self):
        """
        Computes:
            pre_activation_val = sum_s ( z_s @ W_{s->node} )
            z_mu = f( pre_activation_val )
        Returns: None
        """
        # Handle source nodes (in_degree == 0)
        if self.in_degree == 0:
            # Source node! No predictors above; define null prediction
            self.z_mu.zero_()
            return
        # Compute z_mu and pre-activation value a sum of all incoming projections
        if len(self.in_neighbors) > 1:
            for edge, source_node in self.in_neighbors.items():
                a, b = self.weight_map[edge]
                self.inputs_all_concat[:, a:b] = (
                    source_node.z_latent
                )  # Gather all inputs
            self.pre_activation_val = torch.matmul(
                self.inputs_all_concat, self.W
            )  # Compute edge contribution
        else:
            source_node = next(iter(self.in_neighbors.values()))
            self.pre_activation_val = torch.matmul(source_node.z_latent, self.W)
        # Add the biases
        if self.use_bias:
            self.pre_activation_val += self.b
        # Compute predicted state z_mu
        self.z_mu = self.activation_fn(self.pre_activation_val)

    def init_weights(self):
        # Initialize weights
        n_features = sum(
            [source_node.dim for source_node in self.in_neighbors.values()]
        )
        self.W = self.weight_init_std * torch.randn(
            n_features, self.dim, device=self.device
        )
        if self.use_bias:
            self.b = torch.zeros(1, self.dim, device=self.device)

    def setup_optimizer(self, optim_config):
        # Set up optimizers for edges
        self.optimizer = pc_optim.instantiate_optimizer(self.W, optim_config)
        if self.use_bias:
            self.bias_optimizer = pc_optim.instantiate_optimizer(self.b, optim_config)

    def update_weights(self):
        self.optimizer.step(self.compute_weight_grad())
        if self.use_bias:
            self.bias_optimizer.step(self.compute_bias_grad())

    def compute_weight_grad(self):
        if len(self.in_neighbors) == 0:
            return None
        elif len(self.in_neighbors) == 1:
            # single input
            source_node = next(iter(self.in_neighbors.values()))
            grad_st = -torch.matmul(source_node.z_latent.t(), self.gain_mod_error)
        else:
            # multiple inputs
            grad_st = -torch.matmul(self.inputs_all_concat.t(), self.gain_mod_error)
        return grad_st

    def compute_bias_grad(self):
        grad_b = -torch.sum(self.gain_mod_error, dim=0, keepdim=True)
        return grad_b

    def compute_latent_grad(self):
        grad_i = torch.zeros_like(self.z_latent, device=self.device)
        grad_i += self.error  # Add the local error contribution
        for edge, target_node in self.out_neighbors.items():
            tgt_jacob = target_node.get_jacobian(
                edge
            )  # Get the Jacobian of the target node
            # print(f" latent grad for node {self.name} dim {self.dim} via target {target_node.name} dim {target_node.dim} with jacobian {tgt_jacob.shape}")
            grad_i -= torch.matmul(
                target_node.gain_mod_error, tgt_jacob.t()
            )  # Subtract the edge contribution
        return grad_i

    def get_jacobian(self, edge):
        """
        Returns the Jacobian matrix of the target node w.r.t. this node through the given edge.
        For a linear node, this is simply the weight matrix W of the edge.
        The Jacobian has shape [dim_source, dim_target].
        Note: This is the Jacobian of the pre-activation value; to get the Jacobian of the activated state,
        it should be multiplied element-wise by the activation derivative.
        """
        [a, b] = self.weight_map[edge]
        # print(f"node {self.name} weight shapes {self.W.shape} getting jacobian for edge {edge.key()} with indices [{a}, {b}]")
        return self.W[a:b, :].clone()


# Helper functions
def create_node_from_config(config: dict):
    """
    Instantiate a PCNodeBase subclass from a config dictionary.
    The config dictionary must contain a 'type' key specifying the node type,
    and may contain additional keys for parameters required by certain node types.
    Supported node types:
        - 'linear': LinearPCNode
    Returns:
        - node: instance of a PCNodeBase subclass
    Example usage:
        node = get_node_of_type({'type': 'linear', 'name': 'node1', 'dim': 128, 'activation': {'type': 'relu'}})
    """
    if "type" not in config:
        raise ValueError("config['type'] is required")
    type = config["type"].lower()
    config = {
        k: v for k, v in config.items() if k != "type"
    }  # skip the 'type' key in arguments

    if type == "linear":
        return LinearPCNode(config)
    else:
        raise ValueError(f"Unknown node type '{type}'. Supported: 'linear'.")
