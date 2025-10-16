# trainer.py

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.math_utils import MAE, RMSE, MAPE, un_z_score

@torch.no_grad()
def evaluate(model, dataloader, device, mean, std, tag="Val"):
    model.eval()
    mae, rmse, mape, n = 0, 0, 0, 0
    y_pred_list, y_truth_list = [], []

    for batch in dataloader:
        batch = batch.to(device)
        pred = model(batch)
        truth = batch.y.view(pred.shape)

        truth = un_z_score(truth, mean, std)
        pred = un_z_score(pred, mean, std)

        y_pred_list.append(pred)
        y_truth_list.append(truth)

        rmse += RMSE(truth, pred)
        mae += MAE(truth, pred)
        mape += MAPE(truth, pred)
        n += 1

    rmse, mae, mape = rmse / n, mae / n, mape / n
    print(f"[{tag}] MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")
    return rmse, mae, mape, y_pred_list, y_truth_list

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred.float(), batch.y.view(pred.shape).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, config, device, mean, std):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['INITIAL_LR'],
                                 weight_decay=config['WEIGHT_DECAY'])
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_model = None

    for epoch in range(1, config['EPOCHS'] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

        if epoch % 5 == 0:
            _, val_mae, _, _, _ = evaluate(model, val_loader, device, mean, std, tag="Val")
            if val_mae < best_val:
                best_val = val_mae
                best_model = model.state_dict().copy()

    model.load_state_dict(best_model)
    return model

def test_model(model, test_loader, device, config, mean, std, n_node):
    _, _, _, y_pred, y_truth = evaluate(model, test_loader, device, mean, std, tag="Test")

    # --- PLOTTING IMPROVEMENTS START ---

    # Concatenate all batches
    y_pred = torch.cat(y_pred).cpu()
    y_truth = torch.cat(y_truth).cpu()

    # Reshape from (num_samples * num_nodes, N_PRED) to (num_samples, num_nodes, N_PRED)
    num_samples = y_pred.shape[0] // n_node
    y_pred = y_pred.view(num_samples, n_node, config['N_PRED'])
    y_truth = y_truth.view(num_samples, n_node, config['N_PRED'])

    # --- Plotting a single node's predictions over time ---
    node_to_plot = 0  # We'll plot the first sensor
    # We plot the first prediction step (t+1) for each sample
    predictions_for_node = y_pred[:, node_to_plot, 0]
    ground_truth_for_node = y_truth[:, node_to_plot, 0]
    
    # Define the number of time steps to plot
    plot_range = config['N_SLOT'] # Plot one day's worth of data

    plt.figure(figsize=(15, 6))
    
    # Plot Ground Truth in blue
    plt.plot(ground_truth_for_node[:plot_range], label="Ground Truth", color="#3498db", linewidth=2)
    
    # Plot Predictions in red with a dashed line
    plt.plot(predictions_for_node[:plot_range], label="Prediction", color="#e74c3c", linestyle='--', linewidth=2)
    
    plt.title(f"Traffic Speed Prediction vs. Ground Truth for Node {node_to_plot}", fontsize=16)
    plt.xlabel("Time Step (5-minute intervals)", fontsize=12)
    plt.ylabel("Traffic Speed (un-normalized)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("predictions_improved.png", dpi=300)
    plt.show()
    
    # --- PLOTTING IMPROVEMENTS END ---