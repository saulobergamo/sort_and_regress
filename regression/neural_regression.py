import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from util.load_data import training, blind_testing, neural_regression_result

import csv

def neural_regressor():

    # Carrega os dados com label
    with open(training, "r") as training_file:
        labeled_data = [list(map(float, row)) for row in csv.reader(training_file, delimiter=",")]
        labeled_features = [row[3:-2] for row in labeled_data]
        labeled_target = [row[-2] for row in labeled_data]

    # Carrega os dados sem label
    with open(blind_testing, "r") as blind_testing_file:
        unlabeled_data = [list(map(float, row)) for row in csv.reader(blind_testing_file, delimiter=",")]
        data = [row for row in unlabeled_data]
        unlabeled_data = [row[1:] for row in data]

    # Padronizar os dados de treinamento
    scaler = StandardScaler()
    labeled_features = scaler.fit_transform(labeled_features)

    # Treinar o modelo de regressão neural
    regressor = MLPRegressor(hidden_layer_sizes=(32, 64, 128, 64, 32, 8), activation='relu', max_iter=1500,
                            learning_rate='adaptive', learning_rate_init=0.001)
    regressor.fit(labeled_features, labeled_target)

    # Padronizar os dados de teste
    unlabeled_data = scaler.transform(unlabeled_data)

    # Aplicar a regressão nos dados de teste
    y_pred = regressor.predict(unlabeled_data)

    y_pred = [f'{value:.4f}'.replace('.', ',') for value in y_pred]
    # Salvar valores previstos em um arquivo .txt
    with open(neural_regression_result, 'w') as file:
        file.write('\n'.join(y_pred))
