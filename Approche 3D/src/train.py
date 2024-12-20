# train.py

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm  # Importation de tqdm
import matplotlib.pyplot as plt

def train_with_evaluation(model, train_loader, val_loader, num_epochs=25, lr=1e-4, weight_decay=1e-5, patience=5, device='cpu'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    val_aucs = []
    best_model = None
    best_score = -np.inf 
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
        for x, y in train_bar:
            x, y = x.to(device), y.to(device) # move data to device
            optimizer.zero_grad() # reset gradients
            outputs = model(x).squeeze(1) # forward pass
            loss = criterion(outputs, y.float()) # compute loss
            loss.backward() # backward pass
            optimizer.step() # update model parameters

            train_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == y.long()).sum().item()
            total += y.size(0)

            current_acc = correct / total

            train_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{current_acc:.2f}"}) # update progress bar

        train_acc = correct / total
        train_losses.append(train_loss / total)
        train_accuracies.append(train_acc)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_probs, all_labels = [], []

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
        with torch.no_grad(): # disable gradients for validation
            for x, y in val_bar:
                x, y = x.to(device), y.to(device) # move data to device
                outputs = model(x).squeeze(1)
                loss = criterion(outputs, y.float())

                val_loss += loss.item() * x.size(0) # upadate validation loss
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()

                correct += (preds == y.long()).sum().item()
                total += y.size(0)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

                current_val_acc = correct / total

                val_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{current_val_acc:.2f}", 'AUC': 'Calculating...'})

        val_acc = correct / total
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.0  # handle case where AUC cannot be calculated

        val_losses.append(val_loss / total)
        val_accuracies.append(val_acc)
        val_aucs.append(val_auc)

        val_bar.set_postfix({'Loss': f"{val_loss / total:.4f}", 'Acc': f"{val_acc:.2f}", 'AUC': f"{val_auc:.4f}"})

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_losses[-1]:.4f}, Train Acc={train_accuracies[-1]:.4f}, "
              f"Val Loss={val_losses[-1]:.4f}, Val Acc={val_accuracies[-1]:.4f}, Val AUC={val_auc:.4f}")

        compromise_score = val_acc + val_auc - val_losses[-1]

        if compromise_score > best_score:
            best_score = compromise_score
            best_model = model.state_dict() # save the best model state
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience: # stop training if no improvement
            print("Early stopping triggered.")
            break

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "val_aucs": val_aucs
    }
    return best_model, history

def train_without_evaluation(model, train_loader, num_epochs=25, lr=1e-4, weight_decay=1e-5, device='cpu'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
        for x, y in train_bar:
            x, y = x.to(device), y.to(device)  # move data to device
            optimizer.zero_grad()  # reset gradients
            outputs = model(x).squeeze(1)  # forward pass
            loss = criterion(outputs, y.float())  # compute loss
            loss.backward()  # backward pass
            optimizer.step()  # update model parameters

            train_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == y.long()).sum().item()
            total += y.size(0)

            current_acc = correct / total

            train_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{current_acc:.2f}"})  # update progress bar

        train_acc = correct / total
        train_losses.append(train_loss / total)
        train_accuracies.append(train_acc)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_losses[-1]:.4f}, Train Acc={train_accuracies[-1]:.4f}")

    history = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies
    }

    return model.state_dict(), history

