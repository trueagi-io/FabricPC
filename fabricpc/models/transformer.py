"""Deep transformer language-model builder.

Assembles the decomposed transformer (v2) graph — embedding, depth blocks of
MhaResidual → LnMlp1 → Mlp2Residual, and the vocabulary projection:

  tokens → Embedding → MhaResidual(+) → LnMlp1 → Mlp2Residual(+) → VocabProjection → logits
                        │ (skip)   ↑              │ (skip)    ↑
                        └──────────┘              └───────────┘
"""

from typing import Dict, Any, Optional

import jax.numpy as jnp

from fabricpc.nodes import (
    EmbeddingNode,
    MhaResidualNode,
    LnMlp1Node,
    Mlp2ResidualNode,
    VocabProjectionNode,
    Linear,
)
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.core.activations import GeluActivation, IdentityActivation
from fabricpc.core.initializers import (
    NormalInitializer,
    XavierInitializer,
    KaimingInitializer,
)
from fabricpc.core.inference import InferenceBase
from fabricpc.core.mupc import MuPCConfig
from fabricpc.graph_initialization import FeedforwardStateInit


def create_deep_transformer(
    depth: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    seq_len: int,
    vocab_size: int,
    inference: InferenceBase,
    weight_init: Optional[Dict[str, Any]] = None,
):
    """
    Creates a deep transformer graph using the new class-based builder API.

    Note on initialization: the embedding and output-projection nodes
    deliberately OVERRIDE their class-default initializers. The embedding uses
    unit-normal (std=1.0) instead of the node default std=0.02, and the output
    projection uses Normal(std=sqrt(1/embed_dim)) instead of the node default
    Xavier. Both choices keep activations and logits at O(1) variance given that
    muPC scaling is disabled on these two nodes (embedding = discrete lookup,
    output = include_output=False). Changing these without accounting for the
    muPC interaction can cause embedding variance collapse or softmax saturation.
    """
    if weight_init is None:
        # Transformer block weights default to std=0.02 (GPT-style); embedding
        # and output projection set their own init below (see those nodes)
        w_init_obj = NormalInitializer(std=0.02)
    else:
        init_type = weight_init.get("type", "normal")
        if init_type == "normal":
            w_init_obj = NormalInitializer(std=weight_init.get("std", 0.05))
        elif init_type == "xavier":
            w_init_obj = XavierInitializer()
        else:
            w_init_obj = KaimingInitializer()

    nodes = []
    edges = []

    input_node = Linear(
        shape=(seq_len,), activation=IdentityActivation(), name="input_ids"
    )
    nodes.append(input_node)

    # Embedding init: unit-normal (std=1.0), NOT the small std=0.02 used for
    # dense layers. EmbeddingNode is a table lookup with muPC scaling disabled
    # (discrete token indices, not a continuous signal). A Linear+one-hot+muPC
    # path would collapse embedding variance to ~1/vocab_size because muPC
    # assumes dense input with fan_in active features. Unit-normal keeps each
    # token's embedding at O(1) variance going into the first attention block.
    embed_node = EmbeddingNode(
        name="embed",
        shape=(seq_len, embed_dim),
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        weight_init=NormalInitializer(std=1.0),
    )
    nodes.append(embed_node)
    edges.append(Edge(source=input_node, target=embed_node.slot("in")))

    previous_residual = embed_node

    for i in range(depth):
        mha = MhaResidualNode(
            name=f"L{i}_mha",
            shape=(seq_len, embed_dim),
            embed_dim=embed_dim,
            num_heads=num_heads,
            weight_init=w_init_obj,
        )
        nodes.append(mha)
        # previous_residual feeds two edges: "in" is the attention-branch input
        # and is muPC-scaled; "skip" is the residual bypass, unscaled and
        # counted toward residual depth L.
        edges.append(Edge(source=previous_residual, target=mha.slot("in")))
        edges.append(Edge(source=previous_residual, target=mha.slot("skip")))

        mlp1 = LnMlp1Node(
            name=f"L{i}_mlp1",
            shape=(seq_len, mlp_dim),
            embed_dim=embed_dim,
            ff_dim=mlp_dim,
            activation=GeluActivation(),
            weight_init=w_init_obj,
        )
        nodes.append(mlp1)
        edges.append(Edge(source=mha, target=mlp1.slot("in")))

        mlp2 = Mlp2ResidualNode(
            name=f"L{i}_mlp2",
            shape=(seq_len, embed_dim),
            embed_dim=embed_dim,
            ff_dim=mlp_dim,
            weight_init=w_init_obj,
        )
        nodes.append(mlp2)
        edges.append(Edge(source=mlp1, target=mlp2.slot("in")))
        edges.append(Edge(source=mha, target=mlp2.slot("residual")))

        previous_residual = mlp2

    # Output projection init: std = sqrt(1/embed_dim). This keeps pre-softmax
    # logits at O(1) variance regardless of model width — each logit is a dot
    # product over embed_dim features, so scaling by 1/sqrt(embed_dim) prevents
    # logit magnitudes from growing with width. Large initial logits would
    # saturate the softmax and produce near-zero CE gradients at the start of
    # training. muPC is disabled on the output (include_output=False), so this
    # explicit init is what controls output-layer variance.
    logits = VocabProjectionNode(
        name="logits",
        shape=(seq_len, vocab_size),
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        weight_init=NormalInitializer(std=float(jnp.sqrt(1.0 / embed_dim))),
    )
    nodes.append(logits)
    edges.append(Edge(source=previous_residual, target=logits.slot("in")))

    return graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=input_node, y=logits),
        inference=inference,
        scaling=MuPCConfig(include_output=False),
        graph_state_initializer=FeedforwardStateInit(),
    )
