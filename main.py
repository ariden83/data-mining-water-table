import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE


def printHead(name, data):
    print(f"****************** File : {name} ****************")
    print("\n\nLes 5 premières lignes des données :")
    print(data.head())
    print("\nInformations sur les données :")
    print(data.info())


def loadFiles():
    file_path_label = "dataml/training-label.csv"
    file_path_test = "dataml/test-set.csv"
    file_path_value = "dataml/training-value.csv"

    data_label = pd.read_csv(file_path_label)
    printHead("training-label", data_label)

    data_test = pd.read_csv(file_path_test)
    printHead("test-set", data_test)

    data_value = pd.read_csv(file_path_value)
    printHead("training-value", data_value)

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


def encodeVarCat(X, y):
    label_encoders = {}

    # Normaliser la colonne 'permit' pour qu'elle contienne uniquement des booléens
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

    # Encoder y
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)
    return label_encoders, label_encoder_y, X, y


def splitData(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


def trainWithForest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def predict(model, X_val):
    return model.predict(X_val)


def calculPrecision(y_val, y_val_pred):
    precision = accuracy_score(y_val, y_val_pred)
    print(f"Précision trouvée : {precision:.4f}")


def encodeTestData(data_test_cleaned, label_encoders):
    # Gérer la colonne date_recorded avant l'encodage
    if 'date_recorded' in data_test_cleaned.columns:
        data_test_cleaned['date_recorded'] = pd.to_datetime(data_test_cleaned['date_recorded'], errors='coerce')
        data_test_cleaned['date_recorded'] = data_test_cleaned['date_recorded'].fillna(pd.Timestamp(0))
        data_test_cleaned['date_recorded'] = data_test_cleaned['date_recorded'].astype(np.int64) // 10**9

    # Normaliser la colonne 'permit'
    if 'permit' in data_test_cleaned.columns:
        data_test_cleaned['permit'] = data_test_cleaned['permit'].astype(bool)
    if 'public_meeting' in data_test_cleaned.columns:
        data_test_cleaned['public_meeting'] = data_test_cleaned['public_meeting'].astype(bool)

    for column in data_test_cleaned.select_dtypes(include=['object']).columns:
        if column in label_encoders:
            data_test_cleaned[column] = data_test_cleaned[column].fillna('unknown')

            # Remplacer les valeurs non vues par 'unknown'
            unseen_values = set(data_test_cleaned[column]) - set(label_encoders[column].classes_)
            if unseen_values:
                # Remplacer toutes les valeurs non vues par 'unknown'
                data_test_cleaned[column] = data_test_cleaned[column].replace(unseen_values, 'unknown')

            # Transformer avec l'encodeur
            data_test_cleaned[column] = label_encoders[column].transform(data_test_cleaned[column])
        else:
            print(f"Attention : La colonne '{column}' n'a pas été trouvée dans le label_encoder.")

    # Gérer la colonne 'scheme_management' qui pourrait contenir des valeurs numériques
    if 'scheme_management' in data_test_cleaned.columns:
        data_test_cleaned['scheme_management'] = data_test_cleaned['scheme_management'].astype(str)
        data_test_cleaned['scheme_management'] = data_test_cleaned['scheme_management'].fillna('unknown')  # Remplacer les NaN par 'unknown'

    return data_test_cleaned


def predictWithTestData(model, data_test_cleaned):
    return model.predict(data_test_cleaned)


def replaceValuesWithOriginData(label_encoder_y, test_predictions):
    # Remplace les valeurs prédites par les valeurs 'functional', 'non functional'..
    return label_encoder_y.inverse_transform(test_predictions)


def saveFile(data_test, test_predictions_labels):
    output = pd.DataFrame({'id': data_test['id'], 'status_group': test_predictions_labels})
    output.to_csv('SubmissionFormat.csv', index=False)
    print("Sauvegarde effectuée dans 'SubmissionFormat.csv'")

# obsolete
def train_and_evaluate(X, y, features):
    print(f"Évaluation des caractéristiques : {features}")  # Debug
    X_subset = X[list(features)]  # Utilisez la liste des caractéristiques
    X_train, X_val, y_train, y_val = train_test_split(X_subset, y, test_size=0.2, random_state=42)

    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model.fit(X_train, y_train)

    y_val_pred = grid_search.predict(X_val)
    precision = accuracy_score(y_val, y_val_pred)
    return precision

# obsolete
def find_best_features(X, y):
    best_score = 0
    best_features = None

    all_features = X.columns.tolist()
    print(f"Colonnes disponibles dans X : {all_features}")  # Vérifiez les colonnes disponibles

    for i in range(1, len(all_features) + 1):
        for features in itertools.combinations(all_features, i):
            print(f"Test de la combinaison de caractéristiques : {features}")  # Debug

            try:
                score = train_and_evaluate(X, y, features)
            except KeyError as e:
                print(f"Erreur lors de l'accès aux caractéristiques : {e}. Caractéristiques tentées : {features}")  # Afficher l'erreur
                continue  # Passer à la prochaine combinaison

            print(f"Testé les caractéristiques : {features} avec un score de précision : {score:.2f}")

            if score > best_score:
                best_score = score
                best_features = features

    return best_features, best_score


# Chargement des fichiers
data_label, data_value, data_test = loadFiles()
train_data = joinLabels(data_label, data_value)

# Spécifie les colonnes à supprimer
dropColumns = [
    # 'wpt_name', 'region_code', 'lga', 'recorded_by',
    # 'extraction_type_group', 'extraction_type_class',
    # 'payment_type', 'source_type', 'source_class',
    # 'ward', 'subvillage', 'public_meeting',
    # 'scheme_management', 'scheme_name', 'permit'
]

# Nettoyage des données d'entraînement et de test
train_data = cleanTrainData(train_data, dropColumns)
data_test_cleaned = cleanTrainData(data_test, dropColumns)

# Détection des colonnes mixtes et définition de X et y
X = detectMixColumnAndGetTarget(train_data)
y = defineY(train_data)

# X = pd.concat([X, train_data[['year_recorded']]], axis=1)

# Encodage des variables catégorielles
label_encoders, label_encoder_y, X, y = encodeVarCat(X, y)

# Encodage des données de test
data_test_cleaned = encodeTestData(data_test_cleaned, label_encoders)


# Division des données
X_train, X_val, y_train, y_val = splitData(X, y)

# Entraînement du modèle
model = trainWithForest(X_train, y_train)

# Prédiction et calcul de la précision
y_val_pred = predict(model, X_val)
calculPrecision(y_val, y_val_pred)

# Prédiction sur les données de test et sauvegarde des résultats
test_predictions = predictWithTestData(model, data_test_cleaned)
test_predictions_labels = replaceValuesWithOriginData(label_encoder_y, test_predictions)
saveFile(data_test, test_predictions_labels)
