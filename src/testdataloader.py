import argparse
from data.patient_mci_hippocampus_dataset import MCIHippocampusDataset

def test_dataloader(args):
    dataset = MCIHippocampusDataset(
        folder_path=args.data_path,
        csv_file=args.csv_path,
        target_size=(256, 256),
        threshold=args.threshold
    )

    # Vérifier le nombre de patients
    print(f"Total patients MCI détectés: {len(dataset.patient_types)}")

    # Vérifier quelques exemples
    if len(dataset) > 0:
        print("Exemples de données chargées:")
        for i in range(min(5, len(dataset))):
            scan_tensor, label, patient_id = dataset[i]
            print(f"Patient ID: {patient_id}, Label: {label}, Tensor shape: {scan_tensor.shape}")
    else:
        print("Aucune donnée n'a été chargée.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MCI Hippocampus DataLoader")
    parser.add_argument("--data_path", type=str, required=True, help="Chemin vers le dossier des données")
    parser.add_argument("--csv_path", type=str, required=True, help="Chemin vers le fichier CSV")
    parser.add_argument("--threshold", type=float, default=0.15, help="Seuil pour le contenu en hippocampe")
    args = parser.parse_args()

    test_dataloader(args)
