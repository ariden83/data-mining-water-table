import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import plotly.graph_objs as go
import os
import xgboost

dataml_path = os.getenv('DATAML_PATH', './')

print(f"Le chemin du dossier dataml est : {dataml_path}")
print(f"XGBoost version: {xgb.__version__}")
print(f"XGBoost version: {xgboost.__version__}")

def printHead(name, data):
    print(f"****************** File : {name} ****************")
    print("\n\nLes 5 premières lignes des données :")
    print(data.head())
    print("\nInformations sur les données :")
    print(data.info())


def loadFiles():
    file_path_label = dataml_path + "dataml/training-label.csv"
    file_path_test = dataml_path + "dataml/test-set.csv"
    file_path_value = dataml_path + "dataml/training-value.csv"

    data_label = pd.read_csv(file_path_label)
    # printHead("training-label", data_label)

    data_test = pd.read_csv(file_path_test)
    # printHead("test-set", data_test)

    data_value = pd.read_csv(file_path_value)
    # printHead("training-value", data_value)

    return data_label, data_value, data_test


def joinLabels(data_label, data_value):
    d = data_value.merge(data_label, on='id')
    print("\nDonnées fusionnées :\n")
    print(d.info())
    return d


def cleanTrainData(train_data, dropColumns):
    defaultDropColumn = ['id', 'recorded_by']
    dC = defaultDropColumn + dropColumns
    d = train_data.drop(columns=dC)

    # Transformation de date : extraction de l'année et calcul de l'âge de l'eau
    if 'date_recorded' in d.columns and 'construction_year' in d.columns:
        d['year_recorded'] = pd.to_datetime(d['date_recorded']).dt.year
        d['water_age'] = d['year_recorded'] - d['construction_year']
        d['water_age'] = d['water_age'].replace({0: np.nan}).fillna(0)

    numeric_columns = d.select_dtypes(include=[np.number]).columns
    d[numeric_columns] = d[numeric_columns].fillna(d[numeric_columns].median())

    return d


def detectMixColumnAndGetTarget(train_data):
    X = train_data.drop(columns=['status_group'])
    for column in X.columns:
        unique_types = set(type(val) for val in X[column])
        if len(unique_types) > 1:
            X[column] = X[column].astype(str)
    return X


def defineY(train_data):
    return train_data['status_group']


def splitData(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


def trainWithGridSearchCV():
    random_forest_classifier = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=random_forest_classifier, param_grid=param_grid, cv=5, scoring='accuracy', verbose=3)
    return grid_search


def fitWithSmote(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)


def predict(model, X_val):
    # Vérifier si l'attribut best_iteration existe dans le modèle
    if hasattr(model, 'best_iteration'):
        print(f"Prédiction avec early stopping à l'itération {model.best_iteration}")
        return model.predict(X_val, iteration_range=(0, model.best_iteration + 1))
    else:
        print("best_iteration n'existe pas, prédiction avec le modèle complet")
        return model.predict(X_val)


def calculPrecision(y_val, y_val_pred):
    precision = accuracy_score(y_val, y_val_pred)
    print(f"Précision trouvée : {precision:.4f}")


def encodeData(X, y=None, is_test=False):
    label_encoders = {}

    # Gérer la colonne 'date_recorded'
    if 'date_recorded' in X.columns:
        X['date_recorded'] = pd.to_datetime(X['date_recorded'], errors='coerce')
        X['date_recorded'] = X['date_recorded'].fillna(pd.Timestamp(0))
        X['date_recorded'] = X['date_recorded'].astype(np.int64) // 10**9

    # Normaliser la colonne 'permit'
    if 'permit' in X.columns:
        X['permit'] = X['permit'].astype(bool)

    if 'public_meeting' in X.columns:
        X['public_meeting'] = X['public_meeting'].astype(bool)

    # Encoder les colonnes de X
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        # Ajoutez 'unknown' comme classe si elle n'est pas déjà présente
        if 'unknown' not in le.classes_:
            classes = np.append(le.classes_, 'unknown')
            le.classes_ = classes
        label_encoders[column] = le

    # Gérer la colonne 'scheme_management'
    if 'scheme_management' in X.columns:
        X['scheme_management'] = X['scheme_management'].fillna('unknown')
        X['scheme_management'] = X['scheme_management'].astype('category')

    # Si ce n'est pas un ensemble de test, encoder y
    if y is not None:
        label_encoder_y = LabelEncoder()
        y = label_encoder_y.fit_transform(y)
        return label_encoders, label_encoder_y, X, y

    return label_encoders, X  # Assurez-vous que vous renvoyez toujours le DataFrame X ici


def predictWithTestData(model, data_test_cleaned, expected_types):
    # Obtenir les colonnes attendues par le modèle
    expected_columns = model.feature_names_in_  # Utiliser feature_names_in_ à la place de feature_name_

    # Vérifier les colonnes manquantes
    missing_columns = set(expected_columns) - set(data_test_cleaned.columns)
    if missing_columns:
        print(f"Colonnes manquantes dans les données de test : {missing_columns}")
        raise ValueError(f"Les données de test ne contiennent pas toutes les colonnes attendues : {missing_columns}")

    # Vérifier si les colonnes sont dans le bon ordre
    data_test_cleaned = data_test_cleaned[expected_columns]

    # Vérification des types
    for col in expected_columns:
        expected_type = expected_types.get(col)
        actual_type = data_test_cleaned[col].dtype
        if expected_type and str(actual_type) != expected_type:
            print(f"Type de la colonne '{col}' incorrect : {actual_type}, attendu : {expected_type}")
            raise ValueError(f"Type de la colonne '{col}' incorrect")

    return model.predict(data_test_cleaned)



def replaceValuesWithOriginData(label_encoder_y, test_predictions):
    # Remplace les valeurs prédites par les valeurs 'functional', 'non functional'..
    return label_encoder_y.inverse_transform(test_predictions)


def saveFile(data_test, test_predictions_labels):
    output = pd.DataFrame({'id': data_test['id'], 'status_group': test_predictions_labels})
    output.to_csv(dataml_path + 'export/SubmissionFormat.csv', index=False)
    print("Sauvegarde effectuée dans 'SubmissionFormat.csv'")


def caracteristicsImportance(model, X_train):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Extraire les noms des caractéristiques dans l'ordre de leur importance
        feature_names = [X.columns[i] for i in indices]
        importance_values = importances[indices]

        # Créer une figure avec plotly
        fig = go.Figure()

        # Ajouter une trace (un barplot) pour les importances
        fig.add_trace(go.Bar(
            x=feature_names,
            y=importance_values,
            marker=dict(color=importance_values, colorscale='Viridis'),
            text=importance_values,
            hoverinfo="text",
            name="Feature Importances"
        ))

        # Mise en forme de la figure
        fig.update_layout(
            title="Importance des caractéristiques",
            xaxis_title="Caractéristiques",
            yaxis_title="Importance",
            xaxis_tickangle=-90,
            height=600,
            width=800
        )

        fig.show()

    else:
        print("Le modèle n'a pas l'attribut 'feature_importances_'.")


def showGraphic(model):
    results = model.evals_result()

    x_axis = np.arange(0, len(results['validation_0']['mlogloss']))
    train_log_loss = np.array(results['validation_0']['mlogloss'])
    val_log_loss = np.array(results['validation_1']['mlogloss'])

    x_axis_sampled = x_axis[::10]
    train_log_loss_sampled = train_log_loss[::10]
    val_log_loss_sampled = val_log_loss[::10]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_axis_sampled, y=train_log_loss_sampled, mode='lines', name='Train log loss', line_shape='spline'))
    fig.add_trace(go.Scatter(x=x_axis_sampled, y=val_log_loss_sampled, mode='lines', name='Validation log loss', line_shape='spline'))

    early_stopping_point = model.best_iteration
    fig.add_vline(x=early_stopping_point, line=dict(color="black", dash="dash"), annotation_text="Early stopping", annotation_position="top")

    min_val_loss = np.min(val_log_loss)
    min_val_index = np.argmin(val_log_loss)
    fig.add_annotation(x=x_axis[min_val_index], y=min_val_loss,
                       text=f"Min Validation log loss: {min_val_loss:.4f}",
                       showarrow=True, arrowhead=1)

    fig.update_layout(
        title="Évolution du log loss en fonction du nombre d'itérations",
        xaxis_title="Nombre d'itérations",
        yaxis_title="Log Loss",
        template="plotly_white",
        legend=dict(x=0.5, y=1, xanchor='center', orientation='h'),
        hovermode="x"
    )

    fig.show()


def get_model_and_fit(type, X_train, y_train, X_val, y_val):
    # 0.8150
    if type == "RandomForest":
        m = RandomForestClassifier(n_estimators=1000, random_state=42)
        m.fit(X_train, y_train)
        plot_n_estimators_vs_accuracy()
        return m
    # 0.8089
    if type == "RandomForestWithGridSearch":
        m = trainWithGridSearchCV()
        m.fit(X_train, y_train)
        return m

    elif type == "GradientBoosting":
        m = GradientBoostingClassifier(n_estimators=100, random_state=42)
        m.fit(X_train, y_train)
        return m

    # 0.8082
    elif type == "XGBoost":
        m = xgb.XGBClassifier(n_estimators=1000, early_stopping_rounds=50, eval_metric='mlogloss')
        # m = XGBClassifier(n_estimators=1000, random_state=42, use_label_encoder=False, eval_metric='mlogloss')

        eval_set = [(X_train, y_train), (X_val, y_val)]
        m.fit(X_train, y_train, eval_set=eval_set,  verbose=True)

        showGraphic(m)
        return m
    # 0.7942
    elif type == "LightGBM":
        m = LGBMClassifier(n_estimators=100, random_state=42)
        m.fit(X_train, y_train)
        return m
    # 0.8058
    elif type == "CatBoost":
        m = CatBoostClassifier(iterations=10000, task_type="CPU", random_state=42, verbose=0)
        m.fit(X_train, y_train.ravel())
        return m
    else:
        raise ValueError(f"Modèle non supporté : {type}")


def plot_n_estimators_vs_accuracy(X_train, y_train, X_val, y_val, n_estimators_range):
    # Liste pour stocker les précisions
    accuracy_scores = []

    # Boucle sur les différentes valeurs de n_estimators
    for n in n_estimators_range:
        print(f"Entraînement avec n_estimators = {n}")

        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_val_pred)
        accuracy_scores.append(accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, accuracy_scores, marker='o', linestyle='-', color='b')
    plt.title('Précision en fonction de n_estimators pour RandomForestClassifier')
    plt.xlabel('n_estimators')
    plt.ylabel('Précision')
    plt.grid(True)
    plt.show()


model_type = "RandomForest"

dropColumns = [
    # 'wpt_name', 'region_code', 'lga', 'recorded_by',
    # 'extraction_type_group', 'extraction_type_class',
    # 'payment_type', 'source_type', 'source_class',
    # 'ward', 'subvillage', 'public_meeting',
    # 'scheme_management', 'scheme_name', 'permit'
]


data_label, data_value, data_test = loadFiles()
train_data = joinLabels(data_label, data_value)

train_data = cleanTrainData(train_data, dropColumns)
data_test_cleaned = cleanTrainData(data_test, dropColumns)

X = detectMixColumnAndGetTarget(train_data)
y = defineY(train_data)

label_encoders, label_encoder_y, X, y = encodeData(X, y)

_, data_test_cleaned = encodeData(data_test_cleaned, is_test=True)

# Vérification du type de data_test_cleaned
print(type(data_test_cleaned))  # Cela doit être un DataFrame
print(data_test_cleaned.head())  # Affiche les

# Division des données
X_train, X_val, y_train, y_val = splitData(X, y)

# y_train = y_train.ravel()
# y_val = y_val.ravel()

# Smote decrease 0.005
# X_train, y_train = fitWithSmote(X_train, y_train)

model = get_model_and_fit(model_type, X_train, y_train, X_val, y_val)

y_val_pred = predict(model, X_val)
calculPrecision(y_val, y_val_pred)

caracteristicsImportance(model, X_train)

expected_types = {
    'amount_tsh': 'float64',
    'another_feature': 'int64',  # add other features as needed
    # ... add other feature types
}
test_predictions = predictWithTestData(model, data_test_cleaned, expected_types)
test_predictions_labels = replaceValuesWithOriginData(label_encoder_y, test_predictions)
saveFile(data_test, test_predictions_labels)
