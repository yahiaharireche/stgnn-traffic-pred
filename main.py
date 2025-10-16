# main.py

import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from data_loader.dataloader import distance_to_weight, make_dataset
from models.st_gat import ST_GAT
from models.trainer import train_model, test_model

def main():
    config = {
        'BATCH_SIZE': 50,
        'EPOCHS': 50,
        'INITIAL_LR': 3e-4,
        'WEIGHT_DECAY': 5e-5,
        'N_PRED': 9,
        'N_HIST': 12,
        'N_DAY_SLOT': 288,
        'N_DAYS': 44,
        'SPLITS': (34, 5, 5),  # train, val, test days
        'USE_GAT_WEIGHTS': True,
    }

    config['N_SLOT'] = config['N_DAY_SLOT'] - (config['N_PRED'] + config['N_HIST']) + 1

    # Load distance matrix & weight matrix
    distances = pd.read_csv('./dataset/PeMSD7_W_228.csv', header=None).values
    W = distance_to_weight(distances, gat_version=config['USE_GAT_WEIGHTS'])

    # Dataset
    train, val, test, mean, std, n_node = make_dataset(config, W)

    train_loader = DataLoader(train, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val, batch_size=config['BATCH_SIZE'], shuffle=True)
    test_loader = DataLoader(test, batch_size=config['BATCH_SIZE'], shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Model
    model = ST_GAT(config['N_HIST'], config['N_PRED'], n_node, dropout=0.2).to(device)

    # Train & Test
    model = train_model(model, train_loader, val_loader, config, device, mean, std)
    
    # MODIFICATION: Pass n_node to the test_model function
    test_model(model, test_loader, device, config, mean, std, n_node)

if __name__ == "__main__":
    main()