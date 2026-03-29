import os
import pandas as pd
import sys
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    cohen_kappa_score, confusion_matrix
)

###############################################
orb = 'S2indices'
region1 = 'champgane'
region2 = 'Camargue'
year_train = 2021
year_test = 2021

output_dir = f"I:/wetland-classification/Newclassificationolddata/results/XG-short/train{region1}_test{region2}/{orb}/"
os.makedirs(output_dir, exist_ok=True)


# =========================================================
# SAVE METRICS + CONFUSION MATRIX
# =========================================================
def printMeasures(y_pred, y_test, class_names, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=None)
    kappa = cohen_kappa_score(y_test, y_pred)
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    print(f"Overall Accuracy={(100*accuracy):.3f}%, Kappa={kappa:.4f}")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: F1={f1[i]*100:.3f}%, Precision={precision[i]*100:.3f}%, Recall={recall[i]*100:.3f}%")
    print("\nConfusion Matrix (counts):")
    print(cm)

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
    cm_percent_df = pd.DataFrame(cm_percent, index=class_names, columns=class_names)

    excel_path = os.path.join(
        output_dir,
        f"full_{model_name}_temporal_orbit{orb}_{region1}{year_train}_{region2}{year_test}.xlsx"
    )

    with pd.ExcelWriter(excel_path) as writer:
        results.to_excel(writer, sheet_name='Class-wise Metrics', index=False)
        summary.to_excel(writer, sheet_name='Summary', index=False)
        cm_df.to_excel(writer, sheet_name='Confusion Matrix Counts')
        cm_percent_df.to_excel(writer, sheet_name='Confusion Matrix Percent')

    plt.figure(figsize=(7, 5))

    ax = sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.1f',
        cmap='Reds',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 16, "weight": "bold"}
    )

    for text in ax.texts:
        text.set_text(text.get_text() + " %")

    plt.ylabel('True Label', fontsize=18, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold', rotation=0)
    plt.yticks(fontsize=14, fontweight='bold', rotation=90)

    fig_path = os.path.join(
        output_dir,
        f"full_{model_name}_temporal_orbit{orb}_{region1}{year_train}_{region2}{year_test}_percent.png"
    )
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()


# =========================================================
# SAVE SAMPLE-LEVEL OUTPUT FOR MAPPING
# =========================================================
def save_prediction_outputs(
    dataset2,
    y_test,
    y_pred,
    y_proba,
    class_names,
    mask_test,
    model_name
):
    """
    Save one row per sample/polygon for later map creation.
    It automatically saves useful fields if they exist in dataset2.
    """

    n_samples = len(y_test)
    out_df = pd.DataFrame()

    # ---------------------------------
    # Add common ID / attribute fields if present
    # ---------------------------------
    possible_fields = [
        "ID"
    ]

    for field in possible_fields:
        if field in dataset2.files:
            arr = np.array(dataset2[field])

            # keep only 1D fields with same sample length
            if arr.ndim == 1 and len(arr) == n_samples:
                out_df[field] = arr.astype(str) if arr.dtype.kind in {"U", "S", "O"} else arr

    # ---------------------------------
    # True / predicted labels
    # ---------------------------------
    out_df["true_code"] = y_test
    out_df["pred_code"] = y_pred
    out_df["true_class"] = class_names[y_test]
    out_df["pred_class"] = class_names[y_pred]
    out_df["correct"] = (y_test == y_pred).astype(int)

    # ---------------------------------
    # Add class probabilities
    # ---------------------------------
    for i, cls in enumerate(class_names):
        out_df[f"prob_{cls}"] = y_proba[:, i]

    out_df["confidence_max"] = np.max(y_proba, axis=1)

    # ---------------------------------
    # Save selected temporal dates used in test
    # ---------------------------------
    if "date" in dataset2.files:
        all_dates_test = np.array(dataset2["date"])
        selected_dates_test = all_dates_test[mask_test]
        selected_dates_str = ",".join([str(d) for d in selected_dates_test])
        out_df["used_dates"] = selected_dates_str

    # ---------------------------------
    # Save CSV
    # ---------------------------------
    csv_path = os.path.join(
        output_dir,
        f"map_output_{model_name}_temporal_orbit{orb}_{region1}{year_train}_{region2}{year_test}.csv"
    )
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\nMap output saved to:\n{csv_path}")

    # Optional: save only wrong predictions
    wrong_df = out_df[out_df["correct"] == 0].copy()
    wrong_csv_path = os.path.join(
        output_dir,
        f"map_output_{model_name}_WRONG_ONLY_temporal_orbit{orb}_{region1}{year_train}_{region2}{year_test}.csv"
    )
    wrong_df.to_csv(wrong_csv_path, index=False, encoding="utf-8-sig")

    print(f"Wrong-only output saved to:\n{wrong_csv_path}")


# =========================================================
# SETTINGS
# =========================================================
year_combinations = [(2021, 2021)]

dataset_pairs = [
    ("S2indices", "S2indices"),
]


def normalize_dates(dates):
    dates = np.array(dates)
    if np.issubdtype(dates.dtype, np.datetime64):
        return np.array([int(str(d).replace('-', '')[:8]) for d in dates])
    if dates.dtype.type in [np.str_, np.object_]:
        return np.array([int(str(d).replace('-', '').replace('/', '')[:8]) for d in dates])
    return dates.astype(int)


# =========================================================
# MAIN LOOP
# =========================================================
for year_train, year_test in year_combinations:
    model_name = "XGBoost"
    rng_seed = 42
    np.random.seed(rng_seed)

    for train_name, test_name in dataset_pairs:
        name_out = f'full_{model_name}_temporal_orbit{orb}_{region1}{year_train}_{region2}{year_test}_classes.txt'
        output_path = f"{output_dir}/{name_out}"

        original_stdout = sys.stdout
        sys.stdout = open(output_path, "w", encoding="utf-8")

        try:
            test_data_path = 'F:/wetland-classification/Camargue/S2/processed_S2_data-abbrhabitat_Z1_2021_Jan_Dec2.npz'
            train_data_path = 'F:/ecosystem_forest/Wetland_code_Sami/2021/dataset/S2/processed_S2_data-abbrhabitat_Z1_2021_Jan_Dec.npz'

            dataset1 = np.load(train_data_path, allow_pickle=True)
            dataset2 = np.load(test_data_path, allow_pickle=True)

            x_train = dataset1[train_name].astype(np.float64)
            x_test = dataset2[test_name].astype(np.float64)

            dates_train = normalize_dates(dataset1['date'])
            dates_test = normalize_dates(dataset2['date'])

            START_DATE = 20210331
            END_DATE = 20210930

            mask_train = (dates_train >= START_DATE) & (dates_train <= END_DATE)
            mask_test = (dates_test >= START_DATE) & (dates_test <= END_DATE)

            x_train = x_train[:, mask_train, :]
            x_test = x_test[:, mask_test, :]

            print("Selected dates train:", dates_train[mask_train])
            print("Selected dates test:", dates_test[mask_test])
            print("New temporal shape train:", x_train.shape)
            print("New temporal shape test:", x_test.shape)

            # ---------------------------------
            # Normalization
            # ---------------------------------
            p1_train = np.percentile(x_train, 1)
            p99_train = np.percentile(x_train, 99)

            x_train = (x_train - p1_train) / (p99_train - p1_train)
            x_train[x_train > 1] = 1
            x_train[x_train < 0] = 0

            x_test = (x_test - p1_train) / (p99_train - p1_train)
            x_test[x_test > 1] = 1
            x_test[x_test < 0] = 0

            x_train_flattened = x_train.reshape(x_train.shape[0], -1)
            x_test_flattened = x_test.reshape(x_test.shape[0], -1)

            # ---------------------------------
            # Label encoding: SAME mapping for train and test
            # ---------------------------------
            y_train_input = np.array([str(val) for val in dataset1['habitat']])
            y_test_input = np.array([str(val) for val in dataset2['habitat']])

            all_classes = np.unique(np.concatenate([y_train_input, y_test_input]))
            label_to_idx = {label: idx for idx, label in enumerate(all_classes)}

            y_train = np.array([label_to_idx[label] for label in y_train_input])
            y_test = np.array([label_to_idx[label] for label in y_test_input])

            class_names = all_classes

            # ---------------------------------
            # Model
            # ---------------------------------
            xg_model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=len(class_names),
                random_state=rng_seed,
                eval_metric='mlogloss'
            )

            xg_model.fit(x_train_flattened, y_train, verbose=True)

            y_pred_test = xg_model.predict(x_test_flattened)
            y_proba_test = xg_model.predict_proba(x_test_flattened)

            # ---------------------------------
            # Metrics
            # ---------------------------------
            printMeasures(y_pred_test, y_test, class_names, model_name)

            # ---------------------------------
            # NEW PART: save outputs for map creation
            # ---------------------------------
            save_prediction_outputs(
                dataset2=dataset2,
                y_test=y_test,
                y_pred=y_pred_test,
                y_proba=y_proba_test,
                class_names=class_names,
                mask_test=mask_test,
                model_name=model_name
            )

        finally:
            sys.stdout.close()
            sys.stdout = original_stdout

print("Finished.")
