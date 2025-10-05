import torch
import torch.nn.functional as F
from tqdm import tqdm
from evaluation.visualize_experiment import plot_generated_images


# Train Loop
@torch.no_grad()
def train_pcn(
    model,
    data_loader,
    num_epochs,
    eval_callback=None,
    eval_every_epoch=False,
    measure_train_metrics=False,
    iter_callback=None,
):
    # Training metrics
    train_energy_history = []  # Record per-epoch batch-averaged energy trajectories
    eval_metric_history = {
        "train_accuracy": [],
        "val_accuracy": [],
        "train_image_energy": [],
        "val_image_energy": [],
    }

    device = model.device
    x_key = model.task_map["x"]
    y_key = model.task_map["y"]
    y_dim = model.get_dim_for_key(y_key)

    for epoch in tqdm(range(num_epochs)):

        train_epoch_energies = (
            []
        )  # Record batch-averaged energy trajectories for this epoch
        accum_train_acc = torch.tensor(0.0).to(device)
        accum_train_image_energy = torch.tensor(0.0).to(device)
        total = 0

        for x_batch, y_batch in data_loader:
            B = x_batch.size(0)  # Batch length
            batch_energies = (
                []
            )  # Record of batch-averaged energy trajectory for this batch
            total += B

            # Prepare data
            x_batch = x_batch.to(device).view(B, -1)
            y_batch = y_batch.to(device)
            y_onehot = F.one_hot(y_batch, num_classes=y_dim).float()

            # Define clamp layers {task_key: data}
            clamps_train = {x_key: x_batch, y_key: y_onehot}  # input, output

            # Initialize latents
            model.init_latents(clamp_dict=clamps_train, batch_size=B, device=device)
            # Inference
            model.infer(clamps_dict=clamps_train, energy_record=batch_energies)
            # Learning weights
            model.learn(energy_record=batch_energies)

            # Save this batch’s trajectory to this epoch's list
            train_epoch_energies.append(batch_energies)

            # Compute training metrics in minibatch
            if measure_train_metrics:
                # Classification
                clamp = {x_key: x_batch}
                model.init_latents(clamp_dict=clamp, batch_size=B, device=device)
                model.infer(clamps_dict=clamp)
                accum_train_acc += (
                    100 * (model.get_task_result("y").argmax(dim=1) == y_batch).sum()
                )

                # Image generation
                clamp = {y_key: y_onehot}
                model.init_latents(clamp_dict=clamp, batch_size=B, device=device)
                model.infer(clamps_dict=clamp)
                accum_train_image_energy += (
                    0.5 * (model.get_task_result("x") - x_batch).pow(2).sum()
                )

            if iter_callback is not None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                iter_callback()

        # Save this epoch's list of all batch trajectories
        train_energy_history.append(train_epoch_energies)

        # Compute Eval metrics at end of epoch
        if eval_every_epoch and eval_callback is not None:
            acc, engy = eval_callback(model)
        else:
            acc, engy = 0, 0
        eval_metric_history["val_accuracy"].append(acc)
        eval_metric_history["val_image_energy"].append(engy)

        eval_metric_history["train_image_energy"].append(
            accum_train_image_energy.item() / total
        )
        eval_metric_history["train_accuracy"].append(accum_train_acc.item() / total)

    return train_energy_history, eval_metric_history


# Test loop
@torch.no_grad()
def eval_class_accuracy(
    model,
    data_loader,
    analyze_mistakes_func=None,
    plot_generated=False,
    image_size=(1, 28, 28),
):
    """
    Test a PCN in a model-agnostic way.
    Works for both sequential MLP (int-indexed task_map) and graph-based models (name-indexed task_map).
    """
    device = model.device
    x_key = model.task_map["x"]

    total, top1_correct = 0, torch.tensor(0.0).to(device)
    last_x_batch = None

    for x_batch, y_labels in data_loader:
        B = x_batch.size(0)
        total += B

        # Move data to device
        x_batch = x_batch.to(device).view(B, -1)
        y_labels = y_labels.to(device)

        # Run model with input clamped
        image_clamp = {x_key: x_batch}
        model.init_latents(clamp_dict=image_clamp, batch_size=B, device=device)
        model.infer(clamps_dict=image_clamp)
        logits = model.get_task_result("y")  # (B, k_classes)

        # Metrics
        preds1 = logits.argmax(dim=1)
        top1_correct += (preds1 == y_labels).sum()

        last_x_batch = x_batch

    # Optional visualization hook
    if plot_generated:
        try:
            plot_generated_images(
                model, last_x_batch, device=torch.device("cuda"), image_size=image_size
            )
        except NameError:
            pass

    acc = 100 * top1_correct.item() / max(total, 1)
    return acc


@torch.no_grad()
def eval_image_energy(
    model,
    data_loader,
    analyze_mistakes_func=None,
    plot_generated=False,
    image_size=(1, 28, 28),
):
    """
    Test a PCN in a model-agnostic way.
    Works for both sequential MLP (int-indexed task_map) and graph-based models (name-indexed task_map).
    """
    device = model.device
    y_key = model.task_map["y"]
    y_dim = model.get_dim_for_key(y_key)

    count, task_energy = 0, torch.tensor(0.0).to(device)
    last_x_batch = None

    for x_batch, y_batch in data_loader:
        B = x_batch.size(0)
        count += 1

        # Prepare data
        x_batch = x_batch.to(device).view(B, -1)
        y_onehot = F.one_hot(y_batch.to(device), num_classes=y_dim).float()

        # Run model with input clamped
        class_clamp = {y_key: y_onehot}
        model.init_latents(clamp_dict=class_clamp, batch_size=B, device=device)
        model.infer(clamps_dict=class_clamp)
        x_projected = model.get_task_result("x")

        # Metrics
        task_energy += 0.5 * (x_projected - x_batch).pow(2).sum() / B

        last_x_batch = x_batch

    # Optional visualization hook
    if plot_generated:
        try:
            plot_generated_images(
                model, last_x_batch, device=torch.device("cuda"), image_size=image_size
            )
        except NameError:
            pass

    task_energy = task_energy.item() / count
    return task_energy
