import os
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from data_loader_mci import MCIHippocampusDataset
from model2D import BinaryCNN
from torch.utils.data import DataLoader

def weighted_mean(predictions, method="linear", k=5, epsilon=1e-6, threshold=0.1):
    """
    Calcule une moyenne pondérée en fonction de la méthode choisie.
    Args:
        predictions (list): Liste des prédictions des slices.
        method (str): Méthode de pondération.
        k (float): Paramètre pour les méthodes exponentielles/quadratiques.
        epsilon (float): Petite valeur pour éviter les divisions par zéro ou les logs infinis.
        threshold (float): Seuil de confiance pour la méthode "thresholded".
    Returns:
        float: Moyenne pondérée.
    """
    weights = []
    for p in predictions:
        distance = abs(p - 0.5)
        if method == "linear":
            w = distance
        elif method == "quadratic":
            w = distance ** 2
        elif method == "exponential":
            w = 1 - np.exp(-k * distance)
        elif method == "sigmoid":
            w = 1 / (1 + np.exp(-k * (distance - 0.5)))
        elif method == "logarithmic":
            w = -np.log(distance + epsilon)
        elif method == "entropy":
            entropy = -p * np.log(p + epsilon) - (1 - p) * np.log(1 - p + epsilon)
            w = 1 - entropy
        elif method == "quadratic_centered":
            w = distance ** k
        elif method == "thresholded":
            w = 1 if distance > threshold else 0
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        weights.append(w)

    # Calcul de la moyenne pondérée
    weighted_sum = sum(w * p for w, p in zip(weights, predictions))
    total_weight = sum(weights)
    return weighted_sum / total_weight if total_weight > 0 else 0.5


def evaluate_mci(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Charger le modèle
    model = BinaryCNN()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Charger le dataset
    dataset = MCIHippocampusDataset(
        folder_path=args.data_path,
        csv_file=args.csv_path,
        target_size=args.target_size,
        threshold=args.threshold
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Total MCI patients loaded: {len(dataset)}")

    # Obtenir les prédictions pour chaque slice
    patient_slices = {}
    patient_labels = {}

    with torch.no_grad():
        for images, labels, patient_ids in tqdm(loader, desc="Evaluating dataset"):
            images = images.to(device)
            outputs = model(images).squeeze()

            if outputs.dim() == 0:  
                outputs = outputs.unsqueeze(0)  

            for i, patient_id in enumerate(patient_ids):
                if patient_id not in patient_slices:
                    patient_slices[patient_id] = []
                    patient_labels[patient_id] = labels[i].item()

                patient_slices[patient_id].append(outputs[i].item())

    # Méthodes de calcul
    methods = ["mean", "median", "linear", "quadratic", "exponential", "sigmoid", 
               "logarithmic", "entropy", "quadratic_centered", "thresholded"]

    # Calcul des prédictions pour chaque méthode
    patient_predictions = {}
    for patient_id, slice_predictions in patient_slices.items():
        patient_predictions[patient_id] = {}
        
        # Moyenne classique
        patient_predictions[patient_id]["mean"] = sum(slice_predictions) / len(slice_predictions)
        
        # Médiane
        patient_predictions[patient_id]["median"] = np.median(slice_predictions)
        
        # Moyennes pondérées
        for method in methods[2:]:
            patient_predictions[patient_id][method] = weighted_mean(slice_predictions, method=method)

        # Vraie classe
        patient_predictions[patient_id]["true_class"] = patient_labels[patient_id]

    # Calcul des accuracies pour chaque méthode
    accuracies = {}
    for method in methods:
        correct_predictions = sum(
            1 for data in patient_predictions.values()
            if (data[method] > 0.5 and data["true_class"] == 1) or 
               (data[method] <= 0.5 and data["true_class"] == 0)
        )
        total_patients = len(patient_predictions)
        accuracies[method] = correct_predictions / total_patients if total_patients > 0 else 0.0

    # Afficher les accuracies
    print("\nEvaluation complete:")
    for method, acc in accuracies.items():
        print(f"Accuracy ({method.capitalize()}): {acc:.4f}")

    # Sauvegarder les résultats
    results_path = os.path.join(args.save_dir, "mci_stability_results.csv")
    results_df = pd.DataFrame.from_dict(patient_predictions, orient="index")
    results_df.to_csv(results_path)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MCI stability using a binary CNN model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--target_size", type=int, nargs=2, default=(256, 256), help="Target size for slices (height, width)")
    parser.add_argument("--threshold", type=float, default=0.15, help="Threshold for hippocampus content in slices")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save evaluation results")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    evaluate_mci(args)
