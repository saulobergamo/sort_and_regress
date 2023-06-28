import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.tree import plot_tree
from util.load_data import trainning, blind_testing, fuzzy_sort_result
import csv

def fuzzy_gaussian():
    # Carrega os dados de treinamento
    with open(trainning, "r") as training_file:
        trainning_data = list(csv.reader(training_file, delimiter=","))

    # Carrega os dados de teste
    with open(blind_testing, "r") as testing_file:
        testing_data = list(csv.reader(testing_file, delimiter=","))

    # Converte os dados para arrays numpy
    trainning_data = np.array(trainning_data, dtype=float)
    testing_data = np.array(testing_data, dtype=float)

    # Separando os atributos de entrada (x) e o alvo (y) dos dados de treinamento
    x_train = trainning_data[:, :-1]
    y_train = trainning_data[:, -1]

    # Cria as variáveis linguísticas para o atributo qPA
    qPA = ctrl.Antecedent(np.arange(-10, 11), 'qPA')
    qPA['gaussian'] = fuzz.gaussmf(qPA.universe, mean=0, sigma=3)

    # Cria as variáveis linguísticas para o atributo pulso
    pulso = ctrl.Antecedent(np.arange(0, 201), 'pulso')
    pulso['gaussian'] = fuzz.gaussmf(pulso.universe, mean=100, sigma=30)

    # Cria as variáveis linguísticas para o atributo frequência respiratória
    frequencia_respiratoria = ctrl.Antecedent(np.arange(0, 23), 'frequencia_respiratoria')
    frequencia_respiratoria['gaussian'] = fuzz.gaussmf(frequencia_respiratoria.universe, mean=11, sigma=5)

    # Cria a variável linguística para o alvo gravidade
    gravidade = ctrl.Consequent(np.arange(1, 5), 'gravidade')
    gravidade['critico'] = fuzz.gaussmf(gravidade.universe, mean=1, sigma=0.5)
    gravidade['instavel'] = fuzz.gaussmf(gravidade.universe, mean=2, sigma=0.5)
    gravidade['potencialmente_estavel'] = fuzz.gaussmf(gravidade.universe, mean=3, sigma=0.5)
    gravidade['estavel'] = fuzz.gaussmf(gravidade.universe, mean=4, sigma=0.5)

    # Define as regras fuzzy
    rules = [
        ctrl.Rule(qPA['gaussian'] & pulso['gaussian'] & frequencia_respiratoria['gaussian'], gravidade['critico']),
        ctrl.Rule(qPA['gaussian'] & pulso['gaussian'] & frequencia_respiratoria['gaussian'], gravidade['instavel']),
        ctrl.Rule(qPA['gaussian'] & pulso['gaussian'] & frequencia_respiratoria['gaussian'], gravidade['estavel'])
    ]

    # Cria o sistema de controle fuzzy
    gravity_level = ctrl.ControlSystem(rules)
    gravity_level_simulation = ctrl.ControlSystemSimulation(gravity_level)

    with open(fuzzy_sort_result, "w") as f:
        for test_instance in testing_data:
            # Iterar sobre os valores e escrevê-los no arquivo
            gravity_level_simulation.input['qPA'] = test_instance[0]
            gravity_level_simulation.input['pulso'] = test_instance[1]
            gravity_level_simulation.input['frequencia_respiratoria'] = test_instance[2]

            gravity_level_simulation.compute()
            predicted_gravity = gravity_level_simulation.output['gravidade']
            f.write(str(round(predicted_gravity)) + "\n") 
