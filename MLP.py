

import os
import pandas as pd
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix

###############################################
orb = 'band_ind'
region1 = 'champgane'
region2 = 'Camargue'
year_train = 2021
year_test = 2021

output_dir =f"I:/wetland-classification/Newclassificationolddata/results/MLP_short/train{region1}_test{region2}/{orb}/"
os.makedirs(output_dir, exist_ok=True)

def printMeasures(y_pred, y_test, class_names):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=None)
    kappa = cohen_kappa_score(y_test, y_pred)
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    print(f"Overall Accuracy={(100*accuracy):.3f}%, Kappa={kappa:.4f}")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: F1={f1[i]*100:.3f}%, Precision={precision[i]*100:.3f}%, Recall={recall[i]*100:.3f}%")

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
    plt.xticks(fontsize=14, fontweight='bold', rotation=0)
    plt.yticks(fontsize=14, fontweight='bold', rotation=90)

    plt.savefig(
        os.path.join(output_dir,
        f"full_{model_name}_temporal_orbit{orb}_{region1}{year_train}_{region2}{year_test}_percent.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()


year_combinations = [(2021, 2021)]

dataset_pairs = [
        # ("SAR", "SAR"),  # Pair 
             ("band_ind", "band_ind"),  # Pair 1
          # ("S2bands", "S2bands"),  # Pair 1
          # ("S2indices", "S2indices"),  # Pair 1
]

for year_train, year_test in year_combinations:
    model_name = "MLP"
    rng_seed = 42
    np.random.seed(rng_seed)

    for train_name, test_name in dataset_pairs:
        name_out = f'full_{model_name}_temporal_orbit{orb}_{region1}{year_train}_{region2}{year_test}_classes.txt'
        output_path = f"{output_dir}/{name_out}"
        sys.stdout = open(output_path, "w")
        test_data_path = 'F:/wetland-classification/Camargue/S2/processed_S2_data-abbrhabitat_Z1_2021_Jan_Dec2.npz'
        train_data_path  = 'F:/ecosystem_forest/Wetland_code_Sami/2021/dataset/S2/processed_S2_data-abbrhabitat_Z1_2021_Jan_Dec.npz'
        dataset1 = np.load(train_data_path, allow_pickle=True)
        x_train = dataset1[train_name].astype(np.float64)
        dataset2 = np.load(test_data_path, allow_pickle=True)
        x_test = dataset2[test_name].astype(np.float64)

        # ==================================================
        # >>> PERIOD SELECTION BLOCK (ONLY NEW PART)
        # ==================================================

        def normalize_dates(dates):
            dates = np.array(dates)
            if np.issubdtype(dates.dtype, np.datetime64):
                return np.array([int(str(d).replace('-', '')[:8]) for d in dates])
            if dates.dtype.type in [np.str_, np.object_]:
                return np.array([int(str(d).replace('-', '').replace('/', '')[:8]) for d in dates])
            return dates.astype(int)

        dates_train = normalize_dates(dataset1['date'])
        dates_test  = normalize_dates(dataset2['date'])

        START_DATE = 20210331
        END_DATE   = 20210930

        mask_train = (dates_train >= START_DATE) & (dates_train <= END_DATE)
        mask_test  = (dates_test  >= START_DATE) & (dates_test  <= END_DATE)

        x_train = x_train[:, mask_train, :]
        x_test  = x_test[:, mask_test, :]

        print("Selected MLP dates:", dates_train[mask_train])
        print("New MLP temporal shape train:", x_train.shape)

        x_train = (x_train - np.percentile(x_train, 1)) / (np.percentile(x_train, 99) - np.percentile(x_train, 1))
        x_train = np.clip(x_train, 0, 1)

        x_test = (x_test - np.percentile(x_test, 1)) / (np.percentile(x_test, 99) - np.percentile(x_test, 1))
        x_test = np.clip(x_test, 0, 1)

        x_train_flattened = x_train.reshape(x_train.shape[0], -1)
        x_test_flattened = x_test.reshape(x_test.shape[0], -1)

        y_train_input = np.array([str(v) for v in dataset1['habitat']])
        y_test_input = np.array([str(v) for v in dataset2['habitat']])

        unique_classes_train = np.unique(y_train_input)
        label_map = {label: idx for idx, label in enumerate(unique_classes_train)}

        y_train = np.array([label_map[label] for label in y_train_input])
        y_test = np.array([label_map.get(label, -1) for label in y_test_input])

        mlp_model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            max_iter=200,
            batch_size=256,
            learning_rate_init=0.001,
            random_state=rng_seed,
            verbose=True,
            early_stopping=True
        )

        mlp_model.fit(x_train_flattened, y_train)
        y_pred_test = mlp_model.predict(x_test_flattened)

        printMeasures(y_pred_test, y_test, unique_classes_train)
