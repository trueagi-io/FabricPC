import torch
from fabricpc.core.base_pc import PCNet
from fabricpc.core.graph_pc import create_node_from_config, EdgeId


## GRAPH PCN
class PCGraphNet(PCNet):
    """
    Initialize PCGraphNet from a configuration dictionary.

    Expected keys in config:
      - node_list: list[dict], required
      - edge_list: list[dict], required
      - task_map: dict, required
      - device: torch.device or str, optional (default: 'cuda' if available else 'cpu')
      - Optional hyperparameters/flags (if present will be set on the instance):
          T_infer, eta_infer, eta_learn, etc.
    """

    def __init__(self, config: dict):
        super().__init__()
        if "node_list" not in config:
            raise ValueError("config['node_list'] is required")
        if "edge_list" not in config:
            raise ValueError("config['edge_list'] is required")
        if "task_map" not in config:
            raise ValueError("config['task_map'] is required")

        node_list = config["node_list"]
        edge_list = config["edge_list"]
        task_map = config["task_map"]
        device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Core parameters
        self.edge_dictionary = {}  # {source_name->target_name:slot_name: EdgeId object}
        # State variables
        self.node_dictionary = {}  # {name: PCNodeBase object}
        # Hyperparameters defaults
        self.T_infer = 20
        self.eta_infer = 0.1
        self.eta_learn = 1e-3
        self.latent_init_feedforward = True  # Initialize latent state from the projected state if True, else random init
        self.latent_init_std = 0.05  # std for random init of z_latent
        self.init_method = "normal"  # normal or uniform
        self.init_min = 0.0
        self.init_max = 1.0
        # Process configuration
        self.task_map = task_map or {}  # dict {task_name: node_name}
        self.clamp_dict = {}  # dict {node_name: clamped_tensor}
        self.device = device
        self.optimizer = {
            "type": "adam",
            "lr": self.eta_learn,
        }  # default optimizer config

        # Apply any extra config keys to existing attributes
        extra_cfg = {
            k: v
            for k, v in config.items()
            if k not in ("node_list", "edge_list", "task_map", "device")
        }

        if extra_cfg:
            self.set_members(extra_cfg)

        # Build the graph from given node and edge lists
        self.build_graph(node_list, edge_list)

    def build_graph(self, node_list: list, edge_list: list):
        # node_list: list of dictionaries with node parameters
        # edge_list: list of dictionaries with edge parameters
        # Make sure node names are unique
        node_names = [node_params["name"] for node_params in node_list]
        if len(node_names) != len(set(node_names)):
            raise ValueError("Node names must be unique")
        if len(self.node_dictionary) > 0 or len(self.edge_dictionary) > 0:
            raise ValueError("Graph already exists!")

        # Build nodes
        for node_cfg in node_list:
            node = create_node_from_config(node_cfg)  # Create a PCNodeBase object
            node.device = self.device
            self.node_dictionary[node.name] = node  # Add to the node dictionary

        # Build edges
        for edge_config in edge_list:

            edge = EdgeId(
                source=edge_config["source_name"],
                target=edge_config["target_name"],
                slot=edge_config.get("slot", ""),
            )

            if edge.key() in self.edge_dictionary:
                raise ValueError(
                    f"Edge from {edge.source} to {edge.target} slot {edge.slot} is already defined"
                )
            if edge.source == edge.target:
                raise ValueError(
                    f"Self-directed edges are not allowed, node name: {edge.source}"
                )
            if edge.source not in self.node_dictionary:
                raise ValueError(
                    f"Node {edge.source} is not defined in the node dictionary"
                )
            if edge.target not in self.node_dictionary:
                raise ValueError(
                    f"Node {edge.target} is not defined in the node dictionary"
                )

            self.edge_dictionary[edge.key()] = edge  # Add to the edge dictionary

        # Update graph
        for edge in self.edge_dictionary.values():
            # Register the edge with the nodes
            src = self.node_dictionary[edge.source]
            tgt = self.node_dictionary[edge.target]
            src.register_out_neighbor(edge, tgt)
            tgt.register_in_neighbor(edge, src)

        # Initialize nodes
        for node in self.node_dictionary.values():
            node.init_weights()
            node.setup_optimizer(self.optimizer)

    def allocate_node_states(self, batch_size, device=None):
        if device is not None:
            self.device = device
        print(
            f"Allocating node states for batch size {batch_size} on device {self.device}"
        )
        # Allocate tensors once; reuse and fill in-place during inference
        for node in self.node_dictionary.values():
            node.allocate_state(batch_size)

    def init_latents(self, clamp_dict: dict, batch_size: int, device=None):
        if device is not None:
            self.device = device
        self.clamp_dict = clamp_dict

        # Allocate tensors once; reuse and re-fill in-place on subsequent calls
        # check the z_latent of the first node in the dictionary
        n = next(iter(self.node_dictionary.values()))
        need_realloc = (
            (n.z_latent is None)
            or (not isinstance(n.z_latent, torch.Tensor))
            or (n.z_latent.device.type != self.device.type)
            or (n.z_latent.size(0) != batch_size)
        )
        if need_realloc:
            # Fresh allocation (first call or shape/device changed)
            self.allocate_node_states(batch_size, device=self.device)

        for node in self.node_dictionary.values():
            if self.init_method.lower() == "uniform":
                node.z_latent.uniform_(self.init_min, self.init_max)
            elif self.init_method.lower() == "normal":
                node.z_latent.normal_(0, self.latent_init_std)
            else:
                raise ValueError(f"Unknown init method {self.init_method}")

        # Validate clamps and apply
        for name, val in self.clamp_dict.items():
            if name not in self.node_dictionary:
                raise ValueError(f"Clamp key '{name}' not found among graph nodes.")
            node = self.node_dictionary[name]
            if val.dim() != 2 or val.size(1) != node.dim:
                raise ValueError(
                    f"Clamp for node '{name}' must have shape (B, {node.dim}), got {tuple(val.shape)}"
                )
            if val.size(0) != batch_size:
                raise ValueError(
                    f"Clamp for node '{name}' has batch {val.size(0)} != expected {batch_size}"
                )
            # In-place copy of clamped value
            node.z_latent.copy_(val)

        if self.latent_init_feedforward:
            nodes_completed = list(
                self.clamp_dict.keys()
            )  # List of nodes already initialized
            # Feedforward initialization for nodes downstream of any clamped latents
            for name in self.clamp_dict.keys():
                node = self.node_dictionary[name]
                # Traverse the tree below the clamped node. Assumes a tree structure
                self.propagate_init_forward(node.out_neighbors, nodes_completed)
            # Feedforward initialization for any remaining uninitialized nodes (not downstream of any clamp)
            # Start from all source nodes (in_degree == 0)
            for node in self.node_dictionary.values():
                if node.in_degree == 0 and node.name not in nodes_completed:
                    nodes_completed.append(node.name)
                    self.propagate_init_forward(node.out_neighbors, nodes_completed)
            # Check that all nodes have been initialized
            if len(nodes_completed) != len(self.node_dictionary):
                uninit_nodes = set(self.node_dictionary.keys()) - set(nodes_completed)
                raise ValueError(
                    f"Some nodes were not initialized during feedforward init: {uninit_nodes}"
                )

    def propagate_init_forward(self, out_neighbors: dict, nodes_completed: list):
        if len(out_neighbors) == 0:
            return
        # Feedforward initialization
        for edge, node in out_neighbors.items():
            if node.name in nodes_completed:
                continue
            nodes_completed.append(node.name)  # add to list of completed nodes
            # Initialize the node
            node.compute_projection()  # set the node latent state to the projected state of the node
            # Copy the node z_mu values to z_latent
            node.z_latent.copy_(node.z_mu)
            # Process downstream nodes
            self.propagate_init_forward(node.out_neighbors, nodes_completed)

    def update_projections(self):
        """
        Project all nodes to their predicted values (z_mu) based on current latent states of their predictors.
        """
        for node in self.node_dictionary.values():
            node.compute_projection()

    def update_error(self):
        """
        Update prediction errors and gain-modulated errors for all nodes.
        Computes:
            node.error          = z_latent - z_mu
            node.gain_mod_error = error * f'(pre_activation_val)
        Returns: None
        """
        for node in self.node_dictionary.values():
            # Handle source nodes (in_degree == 0)
            if node.in_degree == 0:
                # Source node. No predictors incoming; define zero error.
                node.error.zero_()
                node.gain_mod_error.zero_()
                continue

            # Handle sink nodes (out_degree == 0)
            if node.out_degree == 0:
                # Sink node. No outgoing connections. Check the clamp status.
                if node.name not in self.clamp_dict:
                    # Unclamped sink node, z_latent is undefined, set error to zero.
                    node.error.zero_()
                    node.gain_mod_error.zero_()
                    continue
                # Else, a clamped sink node, compute prediction error normally
            # Else, an internal node, compute prediction error normally
            node.error = node.z_latent - node.z_mu
            node.gain_mod_error = node.error * node.activation_deriv(
                node.pre_activation_val
            )

    def update_latents_step(self):
        # Update latent states for all nodes; one inference step
        for node in self.node_dictionary.values():
            if node.name in self.clamp_dict:
                continue  # Skip clamped nodes
            node.z_latent -= self.eta_infer * node.compute_latent_grad()

    def update_weights(self):
        # Update weights for all edges
        for node in self.node_dictionary.values():
            node.update_weights()

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
        self.update_weights()  # Learning weights step

    def get_task_result(self, task_key):
        # Get the result for a given task from the target node
        if self.task_map is None:
            raise ValueError("task_map not defined")
        node_name = self.task_map.get(task_key, None)
        if node_name is None:
            raise ValueError(f"task_key {task_key} not found in task_map")
        node = self.node_dictionary[node_name]
        if node.out_degree == 0:
            # No latent state available. We get its predicted value by projection.
            return node.z_mu.clone()
        else:
            return node.z_latent.clone()

    def get_dim_for_key(self, key):
        # Graph model: key is a node name
        if key in self.node_dictionary:
            return self.node_dictionary[key].dim
        raise ValueError(f"Unable to resolve dimension for unknown node {key}")

    def get_total_energy(self, energy_record: list, selection_list: list):
        if energy_record is not None:
            B = self.node_dictionary[next(iter(self.node_dictionary))].z_latent.size(
                0
            )  # Batch size for normalization
            total_energy = torch.zeros(1, device=self.device)
            if selection_list is None:
                for node in self.node_dictionary.values():
                    total_energy += 0.5 * node.error.pow(2).sum()
            else:
                for name in selection_list:
                    total_energy += (
                        0.5 * self.node_dictionary[name].error.pow(2).sum()
                    )
            total_energy = total_energy / B
            energy_record.append(total_energy.item())
