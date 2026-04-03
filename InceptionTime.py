
import sys
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from ensemble_main_wetland import trainTestModel
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

###############################################
# AUTOMATIC SETTINGS
###############################################
site_pairs = [
    ("Camargue", "champgane"),
     ("champgane", "Camargue"),
]

dataset_pairs = [
    ("S2bands", "S2bands"),
     ("S2indices", "S2indices"),
     ("band_ind", "band_ind"),
]

year_combinations = [(2021, 2021)]

site_paths = {
    "Camargue":  r"F:/wetland-classification/Camargue/S2/processed_S2_data-abbrhabitat_Camargue_2021_Jan_Dec.npz",
    "champgane": r"F:/wetland-classification/champgane/S2/processed_S2_data-abbrhabitat_champgane_2021_Jan_Dec.npz",

}

###############################################
def printMeasures(y_pred, y_test, output_dir, orb, region1, year_train, region2, year_test, verbose=True):

    # convert torch tensors to numpy if needed
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()

    # === class names from present labels ===
    idx_to_label = {0: 'AV', 1: 'OW', 2: 'TV'}
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    class_names = [idx_to_label[i] for i in unique_labels]

    # === Metrics ===
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=None, labels=unique_labels)
    kappa = cohen_kappa_score(y_test, y_pred)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=unique_labels, zero_division=0
    )

    print(f"Overall Accuracy={(100 * accuracy):.3f}%, Kappa={kappa:.4f}")
    print("F1 Scores:")
    for i, name in enumerate(class_names):
        print(f"{name}: {f1[i] * 100:.3f}%")

    if verbose:
        print("\nPrecision & Recall:")
        for i, name in enumerate(class_names):
            print(f"{name}: Precision={100 * precision[i]:.3f}%, Recall={100 * recall[i]:.3f}%")

        print(f"\nOverall (average): Precision={100 * precision.mean():.3f}%, Recall={100 * recall.mean():.3f}%")

        # === Confusion matrix ===
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

        print("\nConfusion Matrix (counts):")
        print("      " + " ".join(f"{name:>7}" for name in class_names))
        for i, row in enumerate(cm):
            print(f"{class_names[i]:>6}: " + " ".join(f"{val:7d}" for val in row))

        # === Save tables to Excel ===
        results = pd.DataFrame({
            "Class": class_names,
            "F1 Score (%)": f1 * 100,
            "Precision (%)": precision * 100,
            "Recall (%)": recall * 100
        })

        summary = pd.DataFrame({
            "Metric": ["Overall Accuracy", "Kappa"],
            "Value": [accuracy * 100, kappa]
        })

        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

        with pd.ExcelWriter(
            os.path.join(output_dir, f"IT_temporal_orbit{orb}_{region1}{year_train}_{region2}{year_test}.xlsx")
        ) as writer:
            results.to_excel(writer, sheet_name='Class-wise Metrics', index=False)
            summary.to_excel(writer, sheet_name='Summary', index=False)
            cm_df.to_excel(writer, sheet_name='Confusion Matrix')

        # ====================================================
        # HEATMAP
        # ====================================================
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        plt.figure(figsize=(7, 5))
        ax = sns.heatmap(
            cm_percent,
            annot=True,
            fmt='.1f',
            cmap='Reds',
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"size": 26, "weight": "bold"}
        )

        for text in ax.texts:
            text.set_text(text.get_text() + " %")

        plt.ylabel('True Label', fontsize=18, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=18, fontweight='bold')

        plt.xticks(fontsize=22, fontweight='bold', rotation=0)
        plt.yticks(fontsize=22, fontweight='bold', rotation=90)

        plt.savefig(
            os.path.join(
                output_dir,
                f"full_IT_temporal_orbit{orb}_{region1}{year_train}_{region2}{year_test}_percent.png"
            ),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()


def normalize_dates(dates):
    dates = np.array(dates)
    if np.issubdtype(dates.dtype, np.datetime64):
        return np.array([int(str(d).replace('-', '')[:8]) for d in dates])
    if dates.dtype.type in [np.str_, np.object_]:
        return np.array([int(str(d).replace('-', '').replace('/', '')[:8]) for d in dates])
    return dates.astype(int)


def main(argv):
    for region1, region2 in site_pairs:
        for train_name, test_name in dataset_pairs:
            for year_train, year_test in year_combinations:

                orb = train_name
                output_dir = f"I:/wetland-classification/results/IT/train{region1}_test{region2}/{orb}/"
                os.makedirs(output_dir, exist_ok=True)

                model_name = "Inception"
                rng_seed = 42
                show_plots = True
                add_shift = 0
                remove_outliers = False
                check_outliers = False
                outlierType = 'intersect'
                Channeles = True
                n_epochs = 50

                print(f'(Random seed set to {rng_seed})')
                torch.manual_seed(rng_seed)
                np.random.seed(rng_seed)

                sys.stdout = open(
                    f"{output_dir}/full_{model_name}_temporal_orbit{orb}_{region1}{year_train}_{region2}{year_test}.txt",
                    "w",
                    encoding="utf-8"
                )

                # =============================
                # Training data
                # =============================
                dataset = np.load(site_paths[region1], allow_pickle=True)
                y_multi_train = dataset["habitat"]
                x_train = dataset[train_name]
                ID_train = dataset['ID']

                # =============================
                # Test data
                # =============================
                dataset2 = np.load(site_paths[region2], allow_pickle=True)
                y_multi_test = dataset2["habitat"]
                x_test = dataset2[test_name]
                ID_test = dataset2['ID']

                # ==================================================
                # PERIOD SELECTION BLOCK
                # ==================================================
                dates_train = normalize_dates(dataset['date'])
                dates_test = normalize_dates(dataset2['date'])

                START_DATE = int(f"{year_train}0101")
                END_DATE = int(f"{year_train}1231")

                mask_train_dates = (dates_train >= START_DATE) & (dates_train <= END_DATE)
                mask_test_dates = (dates_test >= START_DATE) & (dates_test <= END_DATE)

                x_train = x_train[:, mask_train_dates, :]
                x_test = x_test[:, mask_test_dates, :]

                print("Selected dates train:", dates_train[mask_train_dates])
                print("Selected dates test:", dates_test[mask_test_dates])
                print("New temporal shape train:", x_train.shape)
                print("New temporal shape test:", x_test.shape)

                # ==================================================
                # keep classes existing in train
                # ==================================================
                y_multi_train = np.array([str(v) for v in y_multi_train])
                y_multi_test = np.array([str(v) for v in y_multi_test])

                train_classes = np.unique(y_multi_train)
                sample_mask = np.isin(y_multi_test, train_classes)

                x_test = x_test[sample_mask]
                y_multi_test = y_multi_test[sample_mask]
                ID_test = ID_test[sample_mask]

                # ==================================================
                if Channeles:
                    x_train = x_train
                if Channeles:
                    x_test = x_test

                print(x_train.shape)
                print(x_test.shape)

                x_train = (x_train - np.percentile(x_train, 1)) / (np.percentile(x_train, 99) - np.percentile(x_train, 1))
                x_train[x_train > 1] = 1
                x_train[x_train < 0] = 0

                x_test = (x_test - np.percentile(x_test, 1)) / (np.percentile(x_test, 99) - np.percentile(x_test, 1))
                x_test[x_test > 1] = 1
                x_test[x_test < 0] = 0

                # Pre-process labels: convert to integers
                label_to_idx = {'AV': 0, 'OW': 1, 'TV': 2}

                y_train = np.array([label_to_idx.get(label, -1) for label in y_multi_train])
                y_test = np.array([label_to_idx.get(label, -1) for label in y_multi_test])

                # remove invalid labels if any
                valid_train = y_train != -1
                valid_test = y_test != -1

                x_train = x_train[valid_train]
                y_train = y_train[valid_train]

                x_test = x_test[valid_test]
                y_test = y_test[valid_test]
                ID_test = ID_test[valid_test]

                x_train = torch.Tensor(x_train)
                y_train = torch.LongTensor(y_train)
                x_test = torch.Tensor(x_test)
                y_test = torch.LongTensor(y_test)

                # Permute channel and time dimensions
                x_train = x_train.permute((0, 2, 1))
                x_test = x_test.permute((0, 2, 1))

                # model save path
                in_directory2 = f'F:/wetland-classification/wetland-classification_hard/Newclassificationolddata/results/IT/train{region1}_test{region2}/{orb}'
                os.makedirs(in_directory2, exist_ok=True)
                path = f"{in_directory2}/"
                filename = f'Inceptiontest_orbit{orb}_{n_epochs}ep_{region1}{year_train}_{region2}{year_test}.npz'

                if model_name == "Inception":
                    ensemble_probs = 0
                    for k in range(5):
                        filename_k = filename + f'_{k}'
                        _, y_probs = trainTestModel(model_name, path + filename_k, x_train, x_test, y_train, y_test, n_epochs)
                        ensemble_probs += y_probs / 5
                    y_pred = ensemble_probs.argmax(1)
                    print(f"\n=================================\nENSEMBLE PERFORMANCE\n=================================\n")
                    printMeasures(y_pred, y_test, output_dir, orb, region1, year_train, region2, year_test)
                else:
                    y_pred = trainTestModel(model_name, path + filename, x_train, x_test, y_train, y_test, n_epochs)

                # Debugging prints
                print('#########/ newprintMeasures /#########')
                print(f"type of y_pred: {type(y_pred)}")
                print(f"y_pred content: {y_pred}")
                print(f"y_test shape: {y_test.shape}")
                print(f"y_test content: {y_test}")

                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

                printMeasures(y_pred, y_test, output_dir, orb, region1, year_train, region2, year_test, verbose=True)

                # ==================================================
                # OUTPUT FOR MAP
                # ==================================================
                if isinstance(y_pred, torch.Tensor):
                    y_pred_np = y_pred.cpu().numpy()
                else:
                    y_pred_np = np.array(y_pred)

                if isinstance(y_test, torch.Tensor):
                    y_test_np = y_test.cpu().numpy()
                else:
                    y_test_np = np.array(y_test)

                idx_to_label = {0: 'AV', 1: 'OW', 2: 'TV'}

                df_map = pd.DataFrame({
                    "ID": ID_test,
                    "LULC_true": [idx_to_label[i] for i in y_test_np],
                    "LULC_pred": [idx_to_label[i] for i in y_pred_np],
                })

                if "x" in dataset2.files:
                    df_map["x"] = np.array(dataset2["x"])[sample_mask][valid_test]
                if "y" in dataset2.files:
                    df_map["y"] = np.array(dataset2["y"])[sample_mask][valid_test]
                if "X" in dataset2.files:
                    df_map["X"] = np.array(dataset2["X"])[sample_mask][valid_test]
                if "Y" in dataset2.files:
                    df_map["Y"] = np.array(dataset2["Y"])[sample_mask][valid_test]
                if "nomcomplet" in dataset2.files:
                    df_map["nomcomplet"] = np.array(dataset2["nomcomplet"])[sample_mask][valid_test]

                csv_out = os.path.join(
                    output_dir,
                    f"map_output_IT_orbit{orb}_{region1}{year_train}_{region2}{year_test}.csv"
                )
                df_map.to_csv(csv_out, index=False, encoding="utf-8")
                print("Map output saved:", csv_out)

                np.savez(
                    f'{output_dir}/newy_pred_to_create_map_{filename}.npz',
                    y_pred=y_pred_np,
                    id_parcel_out_test=ID_test,
                    y_multi_test=y_test_np
                )

                sys.stdout.close()
                sys.stdout = sys.__stdout__

    print("All runs completed.")


if __name__ == "__main__":
    main(sys.argv[1:])


# In[ ]:




