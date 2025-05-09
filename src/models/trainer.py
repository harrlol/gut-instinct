import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def train_model(
    model,
    train_loader,
    val_loader,
    compute_r2,
    compute_pearsonr,
    num_epochs=100,
    lr=1e-3,
    weight_decay=0.01,
    patience=10,
    save_plot_path='loss_plot.png',
    criterion=nn.MSELoss(),
    scheduler_patience=5,
    scheduler_factor=0.5,
    verbose=True,
):
    model = model.to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=verbose
    )

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    r2_scores, pearson_scores = [], []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to('cuda'), y_batch.to('cuda')
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        running_val_loss = 0.0
        y_preds, y_trues = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to('cuda'), y_batch.to('cuda')
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                running_val_loss += loss.item()
                y_preds.append(y_pred.cpu().numpy())
                y_trues.append(y_batch.cpu().numpy())

        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)

        y_preds = np.concatenate(y_preds, axis=0)
        y_trues = np.concatenate(y_trues, axis=0)
        r2 = compute_r2(y_trues, y_preds)
        pearson = compute_pearsonr(y_trues, y_preds)
        r2_scores.append(r2)
        pearson_scores.append(pearson)

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R2: {r2:.4f}, Pearson: {pearson:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Best model restored.")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(save_plot_path)
    plt.show()

    return model, {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'r2': r2_scores,
        'pearson': pearson_scores
    }