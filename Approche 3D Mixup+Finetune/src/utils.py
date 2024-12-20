import torch
import matplotlib.pyplot as plt
import numpy as np
import copy

class EarlyStopping:
    """Arrêt anticipé pour prévenir le surapprentissage"""
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): Nombre d'époques sans amélioration avant l'arrêt
            verbose (bool): Si True, affiche un message à chaque fois
            delta (float): Changement minimal considéré comme une amélioration
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.best_model_state = None

    def __call__(self, val_loss, model):
        """
        Vérifie si l'arrêt anticipé doit être déclenché
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} sur {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        """Sauvegarde le modèle quand la validation perd"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss