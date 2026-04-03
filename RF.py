"""

Author: Saeideh Maleki
Date: 2024-11-12

"""
import os
import pandas as pd
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix

###############################################
# AUTOMATIC SETTINGS
###############################################
site_pairs = [
    ("Camargue", "champgane"),
    ("champgane", "Camargue"),
]

year_combinations = [(2021, 2021)]

dataset_pairs = [
    ("S2bands", "S2bands"),
    ("S2indices", "S2indices"),
    ("band_ind", "band_ind"),
]

site_paths = {
    "Camargue":  r"F:/wetland-classification/Camargue/S2/processed_S2_data-abbrhabitat_Camargue_2021_Jan_Dec.npz",
    "champgane": r"F:/wetland-classification/champgane/S2/processed_S2_data-abbrhabitat_champgane_2021_Jan_Dec.npz",

}

def printMeasures(y_pred, y_test, class_names, model_name, output_dir, orb, region1, year_train, region2, year_test):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=None)
    kappa = cohen_kappa_score(y_test, y_pred)
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Convert CM to percentages (row-wise normalization)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    print(f"Overall Accuracy={(100*accuracy):.3f}%, Kappa={kappa:.4f}")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: F1={f1[i]*100:.3f}%, Precision={precision[i]*100:.3f}%, Recall={recall[i]*100:.3f}%")
    print("\nConfusion Matrix (counts):")
    print(cm)

    # Save to Excel
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

    with pd.ExcelWriter(os.path.join(output_dir, f"full_{model_name}_temporal_orbit{orb}_{region1}{year_train}_{region2}{year_test}.xlsx")) as writer:
        results.to_excel(writer, sheet_name='Class-wise Metrics', index=False)
        summary.to_excel(writer, sheet_name='Summary', index=False)
        cm_df.to_excel(writer, sheet_name='Confusion Matrix')

    # ===== CONFUSION MATRIX HEATMAP IN PERCENT =====
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

    # Add % symbol
    for text in ax.texts:
        text.set_text(text.get_text() + " %")

    plt.ylabel('True Label', fontsize=18, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=18, fontweight='bold')

    plt.xticks(fontsize=14, fontweight='bold', rotation=0)
    plt.yticks(fontsize=14, fontweight='bold', rotation=90)

    plt.savefig(
        os.path.join(output_dir, f"full_{model_name}_temporal_orbit{orb}_{region1}{year_train}_{region2}{year_test}_percent.png"),
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()


for region1, region2 in site_pairs:
    for year_train, year_test in year_combinations:
        model_name = "RandomForest"
        rng_seed = 42
        np.random.seed(rng_seed)

        for train_name, test_name in dataset_pairs:
            orb = train_name

            output_dir = f"I:/wetland-classification/results/RF/train{region1}_test{region2}/{orb}/"
            os.makedirs(output_dir, exist_ok=True)

            name_out = f'full_{model_name}_temporal_orbit{orb}_{region1}{year_train}_{region2}{year_test}_classes.txt'
            output_path = f"{output_dir}/{name_out}"
            sys.stdout = open(output_path, "w", encoding="utf-8")

            train_data_path = site_paths[region1]
            test_data_path = site_paths[region2]

            dataset1 = np.load(train_data_path, allow_pickle=True)
            array_names1 = dataset1.files
            print(array_names1)

            x_train = dataset1[train_name].astype(np.float64)

            dataset2 = np.load(test_data_path, allow_pickle=True)
            array_names2 = dataset2.files
            print(array_names2)

            x_test = dataset2[test_name].astype(np.float64)
            x_train = x_train
            x_test = x_test
            print(x_train.shape, x_test.shape)

            #### Normalize
            x_train = (x_train - np.percentile(x_train, 1)) / (np.percentile(x_train, 99) - np.percentile(x_train, 1))
            x_train[x_train > 1] = 1
            x_train[x_train < 0] = 0

            x_test = (x_test - np.percentile(x_test, 1)) / (np.percentile(x_test, 99) - np.percentile(x_test, 1))
            x_test[x_test > 1] = 1
            x_test[x_test < 0] = 0

            #### Flatten
            x_train_flattened = x_train.reshape(x_train.shape[0], -1)
            x_test_flattened = x_test.reshape(x_test.shape[0], -1)

            print(x_train_flattened.shape, x_test_flattened.shape)

            # Labels
            y_train_input = dataset1['habitat']
            y_train_input = np.array([str(val) for val in y_train_input])

            y_test_input = dataset2['habitat']
            y_test_input = np.array([str(val) for val in y_test_input])

            unique_classes_train = np.unique(y_train_input)
            label_to_idx_train = {label: idx for idx, label in enumerate(unique_classes_train)}
            y_train = np.array([label_to_idx_train[label] for label in y_train_input])

            mask = np.isin(y_test_input, unique_classes_train)
            x_test = x_test[mask]
            x_test_flattened = x_test.reshape(x_test.shape[0], -1)
            y_test_input = y_test_input[mask]

            y_test = np.array([label_to_idx_train[label] for label in y_test_input])
            unique_classes_test = unique_classes_train

            # Train RF
            rf_model = RandomForestClassifier(
                n_estimators=100,
                random_state=rng_seed,
                n_jobs=-1
            )
            rf_model.fit(x_train_flattened, y_train)

            # Predict
            y_pred_test = rf_model.predict(x_test_flattened)
            printMeasures(
                y_pred_test, y_test, unique_classes_test, model_name,
                output_dir, orb, region1, year_train, region2, year_test
            )

            # =====================================================
            # OUTPUT FOR MAP
            # =====================================================
            idx_to_label = {idx: label for label, idx in label_to_idx_train.items()}

            df_map = pd.DataFrame({
                "ID": np.array(dataset2["ID"])[mask] if "ID" in dataset2.files else np.arange(len(y_test)),
                "LULC_true": [idx_to_label[i] for i in y_test],
                "LULC_pred": [idx_to_label[i] for i in y_pred_test],
            })

            if "x" in dataset2.files:
                df_map["x"] = np.array(dataset2["x"])[mask]
            if "y" in dataset2.files:
                df_map["y"] = np.array(dataset2["y"])[mask]
            if "X" in dataset2.files:
                df_map["X"] = np.array(dataset2["X"])[mask]
            if "Y" in dataset2.files:
                df_map["Y"] = np.array(dataset2["Y"])[mask]
            if "nomcomplet" in dataset2.files:
                df_map["nomcomplet"] = np.array(dataset2["nomcomplet"])[mask]

            df_map.to_csv(
                os.path.join(output_dir, f"map_output_{model_name}_orbit{orb}_{region1}{year_train}_{region2}{year_test}.csv"),
                index=False,
                encoding="utf-8"
            )
            print("Map output saved.")

            # === Feature Importances ===
            importances = rf_model.feature_importances_

            n_dates = x_train.shape[1]
            n_features = x_train.shape[2]
            importances_matrix = importances.reshape(n_dates, n_features)

            dates = dataset1['date']
            print("Dates shape:", dates.shape)

            feature_names = [f"F{i+1}" for i in range(n_features)]
            importance_df = pd.DataFrame(importances_matrix, index=dates, columns=feature_names)

            # Save all importances
            importance_df.to_excel(os.path.join(output_dir, f"feature_importance_{model_name}_orbit{orb}_{region1}{year_train}_{region2}{year_test}.xlsx"))

            # === Top 20% ===
            top_fraction = 0.2
            n_total = importance_df.size
            n_top = int(np.ceil(top_fraction * n_total))
            importance_long = importance_df.reset_index().melt(
                id_vars="index", var_name="Feature", value_name="Importance"
            ).rename(columns={"index": "Date"})
            importance_long_sorted = importance_long.sort_values(by="Importance", ascending=False)
            top_20_percent = importance_long_sorted.head(n_top)

            top_20_percent.to_excel(
                os.path.join(output_dir, f"top20pct_feature_importance_{model_name}_orbit{orb}_{region1}{year_train}_{region2}{year_test}.xlsx"),
                index=False
            )
            print(f"Saved top {top_fraction*100:.0f}% most important features ({n_top} out of {n_total}).")

            # === Plot heatmap ===
            plt.figure(figsize=(12, 6))
            sns.heatmap(importance_df, cmap="YlOrRd", annot=False)
            plt.title("Feature Importance over Time (RF)")
            plt.ylabel("Date")
            plt.xlabel("Feature")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"feature_importance_heatmap_{model_name}_orbit{orb}_{region1}{year_train}_{region2}{year_test}.png"), dpi=300)
            plt.show()

            sys.stdout.close()
            sys.stdout = sys.__stdout__

print("All runs completed.")


# In[ ]:




