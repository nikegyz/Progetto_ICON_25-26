import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except Exception:  # IPython potrebbe non essere installato in ambiente non notebook
    def display(obj):
        print(obj)

from logical_classes import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, \
    recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB


def construct_confusion_matrix(matrix, display_labels, title):
    # Visualizza la matrice di confusione creata dalla funzione confusion_matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=display_labels)
    disp.plot(cmap='YlGnBu')

    # Rimuovi le linee della griglia delle celle
    plt.grid(which='major')

    plt.title(title)

    # Regola la dimensione del grafico
    fig = plt.gcf()
    fig.set_size_inches(10, 7)

    # Mostra il grafico in una finestra reale (chiudila per proseguire).
    # NB: non chiamare fig.clf() dopo plt.show(): con il backend TkAgg, una volta
    # chiusa la finestra la toolbar è già distrutta e clf() andrebbe in errore
    # (_tkinter.TclError: invalid command name ...). plt.close() chiude in sicurezza.
    plt.show()
    plt.close(fig)



def classification(dataset_path) -> list[ClassificationResult]:
    # Esegue il compito di classificazione per ogni modello implementato
    classification_results = []

    warnings.filterwarnings('ignore')

    pd.set_option('display.max_columns', None)

    def train_evaluate_model(test_set, predictions, prediction_proba, model_name):
        # Calcola le metriche per la valutazione del modello di classificazione
        accuracy = accuracy_score(test_set, predictions)
        f1 = f1_score(test_set, predictions, average='micro')
        precision = precision_score(test_set, predictions, average='micro')
        recall = recall_score(test_set, predictions, average='micro')
        balanced_accuracy = balanced_accuracy_score(test_set, predictions)
        # Per dataset binari (come diabete), usa AUC semplice; per multiclasse (come Iris), usa 'ovr'
        if dataset_path in ['./dataset/diabetes.csv', './dataset/indian_liver_patient.csv']:
            auc = roc_auc_score(test_set, prediction_proba)
        else:
            auc = roc_auc_score(test_set, prediction_proba, multi_class='ovr')

        classification_results.append(
            ClassificationResult(model_name, accuracy, f1, precision, recall,
                                 balanced_accuracy, auc, None, dataset_path)
        )

        evaluated_dataframe = pd.DataFrame(
            [[accuracy, f1, precision, recall, balanced_accuracy, auc]],
            columns=['accuracy', 'f1_score', 'precision', 'recall', 'balanced_accuracy', 'auc']
        )

        return evaluated_dataframe

    print("\n\n=========================================== DATI DEL PROGETTO ==============================================")

    print(f"DEBUG: Loading dataset from: {dataset_path}")
    
    try:
        raw_file = pd.read_csv(dataset_path)
    except FileNotFoundError as e:
        print(f"ERROR: Dataset file not found: {dataset_path}")
        print(f"Error details: {e}")
        raise
    
    print(raw_file.info())

    # Select features for Indian Liver Patient dataset
    # Handle both Gender and non-Gender columns
    available_cols = raw_file.columns.tolist()
    print(f"\nDEBUG: Available columns: {available_cols}")
    
    feature_cols = ["Age", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase", 
                    "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Protiens",  # Note: CSV has typo "Protiens"
                    "Albumin", "Albumin_and_Globulin_Ratio"]
    
    # Filter to only available columns
    feature_cols = [col for col in feature_cols if col in available_cols]
    print(f"DEBUG: Selected columns: {feature_cols}")
    
    if "Dataset" in available_cols:
        dataset = raw_file[feature_cols + ["Dataset"]].copy()
    else:
        dataset = raw_file[feature_cols].copy()
        dataset['Dataset'] = 1  # Default value if not present

    # Convert Dataset values: 1->1 (patient), 2->0 (non-patient)
    dataset['Dataset'] = dataset['Dataset'].apply(lambda x: 1 if x == 1 else 0)
    
    # Separa la variabile target
    y = dataset['Dataset']  # Assegna a y solo la feature di output 'Dataset'
    x = dataset.loc[:, dataset.columns != 'Dataset']  # Assegna a X le features di input
    
    # Pulizia: riempi i valori mancanti con la media
    x = x.fillna(x.mean())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    # -----------------
    # K Nearest Neighbor
    # -----------------
    print('\n----------------------------------------- Allenamento con KNN -----------------------------------------')

    # Definisci l'intervallo dei parametri
    knn_param_grid = {
        'n_neighbors': range(1, 10),
        'weights': ['distance', 'uniform']
    }

    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, refit=True)

    # Adatta il modello per la ricerca a griglia
    knn_grid.fit(x_train, y_train)

    # Stampa i migliori parametri
    print('\nMigliori parametri KNN:\n', knn_grid.best_params_)

    # Predici con i migliori parametri
    knn_target_test_prd = knn_grid.predict(x_test)
    if dataset_path in ['./dataset/diabetes.csv', './dataset/indian_liver_patient.csv']:
        knn_target_test_prd_proba = knn_grid.predict_proba(x_test)[:, 1]
    else:
        knn_target_test_prd_proba = knn_grid.predict_proba(x_test)

    # Costruisci la matrice di confusione
    knn_conf_matrix = confusion_matrix(y_test, knn_target_test_prd, labels=knn_grid.classes_)
    construct_confusion_matrix(knn_conf_matrix, ['Non Liver Patient', 'Liver Patient'], 'K Nearest Neighbor')

    # -----------------
    # Decision Tree
    # -----------------
    print('\n----------------------------------------- Allenamento con DT ------------------------------------------')

    dt = DecisionTreeClassifier(random_state=42)
    dt = dt.fit(x_train, y_train)

    # Definisci l'intervallo dei parametri
    dt_param_grid = {
        'max_depth': range(1, dt.get_depth() + 1, 2),
        'max_features': range(1, len(dt.feature_importances_) + 1)
    }
    dt_grid = GridSearchCV(dt, dt_param_grid, n_jobs=-1)

    # Adatta il modello per la ricerca a griglia
    dt_grid.fit(x_train, y_train)

    # Stampa i migliori parametri
    print('\nMigliori parametri DT:\n', dt_grid.best_params_)

    # Predici con i migliori parametri
    dt_target_test_prd = dt_grid.predict(x_test)
    if dataset_path in ['./dataset/diabetes.csv', './dataset/indian_liver_patient.csv']:
        dt_target_test_prd_proba = dt_grid.predict_proba(x_test)[:, 1]
    else:
        dt_target_test_prd_proba = dt_grid.predict_proba(x_test)

    # Costruisci la matrice di confusione
    dt_conf_matrix = confusion_matrix(y_test, dt_target_test_prd, labels=dt_grid.classes_)
    construct_confusion_matrix(dt_conf_matrix, ['Non Liver Patient', 'Liver Patient'], 'Decision Tree')

    # -----------------
    # Random Forest
    # -----------------
    print('\n----------------------------------------- Allenamento con RF ------------------------------------------')

    rf = RandomForestClassifier(
        random_state=42
    )

    # Definisci l'intervallo dei parametri
    rf_param_grid = {
        'n_estimators': [15, 20, 30, 40],
        'max_depth': range(1, 10),
    }
    rf_grid = GridSearchCV(rf, rf_param_grid, n_jobs=-1)

    # Adatta il modello per la ricerca a griglia
    rf_grid.fit(x_train, y_train)

    # Stampa i migliori parametri
    print('\nMigliori parametri RF:\n', rf_grid.best_params_)

    # Predici con i migliori parametri
    rf_target_test_prd = rf_grid.predict(x_test)
    if dataset_path in ['./dataset/diabetes.csv', './dataset/indian_liver_patient.csv']:
        rf_target_test_prd_proba = rf_grid.predict_proba(x_test)[:, 1]
    else:
        rf_target_test_prd_proba = rf_grid.predict_proba(x_test)

    # Costruisci la matrice di confusione
    rf_conf_matrix = confusion_matrix(y_test, rf_target_test_prd, labels=rf_grid.classes_)
    construct_confusion_matrix(rf_conf_matrix, ['Non Liver Patient', 'Liver Patient'], 'Random Forest')

    # -----------------
    # Naive Bayes
    # -----------------
    print('\n----------------------------------------- Allenamento con NB ------------------------------------------')

    nb = GaussianNB()

    # Allenamento del modello
    nb.fit(x_train, y_train)

    # Predici con i parametri
    nb_target_test_prd = nb.predict(x_test)
    if dataset_path in ['./dataset/diabetes.csv', './dataset/indian_liver_patient.csv']:
        nb_target_test_prd_proba = nb.predict_proba(x_test)[:, 1]
    else:
        nb_target_test_prd_proba = nb.predict_proba(x_test)

    print('\nNessun parametro migliore necessario')

    # Costruisci la matrice di confusione
    nb_conf_matrix = confusion_matrix(y_test, nb_target_test_prd, labels=nb.classes_)
    construct_confusion_matrix(nb_conf_matrix, ['Non Liver Patient', 'Liver Patient'], 'Naive Bayes')

    # -----------------
    # Neural Network
    # -----------------
    print('\n----------------------------------------- Allenamento con NN ------------------------------------------')

    sc = MinMaxScaler()
    scaler = sc.fit(x_train)

    # Utilizzando MinMaxScaler, normalizziamo l'intervallo dei valori nei nostri dati
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Definisci la griglia dei parametri
    param_grid = {
        'hidden_layer_sizes': [(4, 4, 4)],
        'max_iter': [2000, 2100],
        'activation': ['relu'],
        'solver': ['lbfgs'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

    mlp_clf = MLPClassifier(random_state=42)

    nn_grid = GridSearchCV(mlp_clf, param_grid, n_jobs=-1)

    # Adatta il modello per la ricerca a griglia
    nn_grid.fit(x_train_scaled, y_train)

    # Stampa i migliori parametri
    print('\nMigliori parametri NN:\n', nn_grid.best_params_)

    # Predici con i migliori parametri
    nn_target_test_prd = nn_grid.predict(x_test_scaled)
    if dataset_path in ['./dataset/diabetes.csv', './dataset/indian_liver_patient.csv']:
        nn_target_test_prd_proba = nn_grid.predict_proba(x_test_scaled)[:, 1]
    else:
        nn_target_test_prd_proba = nn_grid.predict_proba(x_test_scaled)

    # Costruisci la matrice di confusione
    nn_conf_matrix = confusion_matrix(y_test, nn_target_test_prd, labels=nn_grid.classes_)
    construct_confusion_matrix(nn_conf_matrix, ['Non Liver Patient', 'Liver Patient'], 'Neural Network')

    # -----------------
    # Logistic Regression
    # -----------------
    print('\n--------------------------------------- Costruzione classificatore LR ----------------------------------------')

    sc = StandardScaler()
    scaler = sc.fit(x_train)

    # Utilizzando StandardScaler, normalizziamo l'intervallo dei valori nei nostri dati
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Definisci la griglia dei parametri
    parameters = {
        'penalty': ['l2'],
        'C': np.logspace(-3, 3, 7),
        'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    }

    logreg = LogisticRegression()
    lr_grid = GridSearchCV(logreg, param_grid=parameters, cv=5)

    # Adatta il modello per la ricerca a griglia
    lr_grid.fit(x_train_scaled, y_train)

    # Stampa i migliori parametri
    print('\nMigliori parametri LR:\n', lr_grid.best_params_)

    # Predici con i migliori parametri
    lr_target_test_prd = lr_grid.predict(x_test_scaled)
    if dataset_path in ['./dataset/diabetes.csv', './dataset/indian_liver_patient.csv']:
        lr_target_test_prd_proba = lr_grid.predict_proba(x_test_scaled)[:, 1]
    else:
        lr_target_test_prd_proba = lr_grid.predict_proba(x_test_scaled)

    # Costruisci la matrice di confusione
    lr_conf_matrix = confusion_matrix(y_test, lr_target_test_prd, labels=lr_grid.classes_)
    construct_confusion_matrix(lr_conf_matrix, ['Non Liver Patient', 'Liver Patient'], 'Logistic Regression')

    # -----------------
    # K-Means
    # -----------------
    print('\n---------------------------------------- Predizione con KMeans ----------------------------------------')

    sc = MinMaxScaler()
    scaler = sc.fit(x_train)

    # Utilizzando MinMaxScaler, normalizziamo l'intervallo dei valori nei nostri dati
    x_scaled = scaler.transform(x)

    scaled_dataframe = pd.DataFrame(x_scaled, columns=x.columns)

    # Per il dataset del diabete, usa 2 cluster (non diabetico vs diabetico)
    if dataset_path in ['./dataset/diabetes.csv', './dataset/indian_liver_patient.csv']:
        n_cluster = 2
    else:
        n_cluster = 3

    kmeans_model = KMeans(n_clusters=n_cluster, n_init=10, random_state=42)

    # Allenamento del modello
    kmeans_model.fit(scaled_dataframe)

    scaled_dataframe['cluster'] = kmeans_model.labels_

    # ---- Visualizzazione grafica del K-Means (finestra reale) ----
    labels = kmeans_model.labels_

    # Riduzione a 2 dimensioni con PCA per poter disegnare i cluster su un piano
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x_scaled)

    fig_km, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter dei cluster
    scatter = ax_scatter.scatter(coords[:, 0], coords[:, 1], c=labels,
                                 cmap='YlGnBu', s=18, alpha=0.8, edgecolors='none')
    # Centroidi proiettati nello spazio PCA
    centers_pca = pca.transform(kmeans_model.cluster_centers_)
    ax_scatter.scatter(centers_pca[:, 0], centers_pca[:, 1],
                       c='red', marker='X', s=180, edgecolors='black', label='Centroidi')
    ax_scatter.set_title('K-Means: cluster (proiezione PCA 2D)')
    ax_scatter.set_xlabel('Componente principale 1')
    ax_scatter.set_ylabel('Componente principale 2')
    ax_scatter.legend(loc='best')
    fig_km.colorbar(scatter, ax=ax_scatter, label='Cluster')

    # Barre con la numerosità di ciascun cluster
    counts_series = pd.Series(labels).value_counts().sort_index()
    ax_bar.bar([f'Cluster {i}' for i in counts_series.index],
               counts_series.values, color='#3690c0')
    ax_bar.set_title('Numerosità dei cluster')
    ax_bar.set_ylabel('Numero di campioni')
    for i, v in enumerate(counts_series.values):
        ax_bar.text(i, v, str(v), ha='center', va='bottom')

    fig_km.suptitle('K-Means Clustering', fontweight='bold')
    fig_km.tight_layout()
    plt.show()
    plt.close(fig_km)

    kmeans_counts = pd.Series(scaled_dataframe['cluster']).value_counts(ascending=True).tolist()
    kmeans_score_str = "\n".join(map(str, kmeans_counts))
    classification_results.append(
        ClassificationResult('k_means', None, None, None, None, None, None,
                             kmeans_score_str, dataset_path)
    )

    # Stampa i risultati della classificazione
    print('\n', pd.Series(scaled_dataframe['cluster']).value_counts(sort=True))

    print('\n----------------------------------------------- Risultati ------------------------------------------------')

    knn_results = train_evaluate_model(y_test, knn_target_test_prd, knn_target_test_prd_proba, 'knn')
    knn_results.index = ['K Nearest Neighbors']

    dt_results = train_evaluate_model(y_test, dt_target_test_prd, dt_target_test_prd_proba, 'decision_trees')
    dt_results.index = ['Decision Trees']

    rf_results = train_evaluate_model(y_test, rf_target_test_prd, rf_target_test_prd_proba, 'random_forest')
    rf_results.index = ['Random Forest']

    nb_results = train_evaluate_model(y_test, nb_target_test_prd, nb_target_test_prd_proba, 'naive_bayes')
    nb_results.index = ['Naive Bayes']

    nn_results = train_evaluate_model(y_test, nn_target_test_prd, nn_target_test_prd_proba, 'neural_network')
    nn_results.index = ['Neural Network']

    lr_results = train_evaluate_model(y_test, lr_target_test_prd, lr_target_test_prd_proba, 'logistic_regression')
    lr_results.index = ['Logistic Regression']

    # Lista di tutti i dataframe creati
    frames = [knn_results, dt_results, rf_results, nb_results, nn_results, lr_results]

    # Trasforma la lista in una singola tabella
    results = pd.concat(frames)

    display(results)

    # ---- Tabella riassuntiva delle metriche (finestra reale) ----
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    rounded = results.round(4)
    the_table = ax.table(rowLabels=rounded.index.tolist(),
                         cellText=rounded.values.tolist(),
                         colLabels=rounded.columns.tolist(), loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 1.5)
    ax.set_title('Confronto delle metriche dei modelli', fontweight='bold', pad=20)

    # Massimizza la finestra se il backend lo consente
    try:
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # type: ignore
    except Exception:
        pass  # Backend che non supporta lo zoom: ignora

    # Mostra la tabella in una finestra reale (niente input() bloccante)
    plt.show()
    plt.close(fig)

    return classification_results
