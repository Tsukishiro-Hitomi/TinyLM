"""Language-model training loop."""

import torch
import torch.nn as nn
import numpy.typing as npt
import time
from pathlib import Path

from .utils import save_checkpoint, ExperimentLogger, plot_loss_curve, learning_rate_schedule
from .data import get_batch

run_dir = Path("runs")

def train(train_data: npt.NDArray,
          val_data: npt.NDArray,
          batch_size: int,
          context_length: int,
          device: str,
          iteration: int,
          model: nn.Module,
          loss_function, 
          optimizer: torch.optim.Optimizer,
          iteration_save: int,
          iteration_evaluate: int,
          ):
    print("====== START TRAINING =======")
    model = model.to(device)
    assert model.device == device, f"model should be trained on {device} instead of {model.device}"

    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / time.strftime("%Y%m%d-%H%M%S")
    output_path.mkdir(exist_ok=True)

    checkpoints_path = output_path / "checkpoints"
    checkpoints_path.mkdir(exist_ok=True)

    logger = ExperimentLogger(output_path / "metrics.csv")
    max_lr = optimizer.param_groups[0]["lr"]
    min_lr = max_lr * 0.1
    warmup_steps = max(1, int(iteration * 0.05))

    val_x, val_y = get_batch(val_data, batch_size, context_length, device)

    for i in range(1, iteration + 1):
        lr = learning_rate_schedule(i, max_lr, min_lr, warmup_steps, iteration)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        x, y = get_batch(train_data, batch_size, context_length, device)
        logits = model(x)
        loss = loss_function(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]

        if i % iteration_save == 0:
            checkpoint_path = checkpoints_path / f"checkpoint{i}"
            save_checkpoint(model, optimizer, i, checkpoint_path)
            print(f"iteration{i}: train loss={loss}, checkpoint is saved to {checkpoint_path}.")

        if i % iteration_evaluate == 0:
            val_loss = evaluate(val_x, val_y, model, loss_function)
            logger.log(
                step=i, 
                train_loss = loss, 
                val_loss = val_loss, 
                lr=lr,
            )

    save_checkpoint(model, optimizer, iteration, checkpoints_path / f"checkpoint{iteration}")
    plot_loss_curve(output_path / "metrics.csv", output_path / "loss_curve.png")
    logger.close()
    print("======== TRAINING COMPLETE =======")

@torch.no_grad()
def evaluate(val_x, val_y, model, loss_function):
    model.eval()
    logits = model(val_x)
    return loss_function(logits, val_y)

def main(train_data,
         val_data,
         batch_size,
         context_length,
         device,
         iteration,
         model,
         loss_function,
         optimizer,
         iteration_save,
         iteration_evaluate
         ):
    
    train(train_data, val_data, batch_size, context_length, device, iteration, model, loss_function, optimizer, iteration_save, iteration_evaluate)

if __name__ == "__main__":
    main()
