from sklearn.tree import DecisionTreeClassifier, plot_tree
import csv
import matplotlib.pyplot as plt
from util.load_data import training, blind_testing

def tree_sort():
    
    labeled_data = None

    # Carrega os dados com label
    with open(training, "r") as training_file:
        labeled_data = [list(map(float, row)) for row in csv.reader(training_file, delimiter=",")]
        labeled_features = [row[3:-2] for row in labeled_data]
        labeled_target = [row[-1] for row in labeled_data]

    # Carrega os dados sem label
    with open(blind_testing, "r") as blind_testing:
        unlabeled_data = [list(map(float, row)) for row in csv.reader(blind_testing, delimiter=",")]
        data = [row for row in unlabeled_data]
        unlabeled_data = [row[1:] for row in data]
        validate = [row[-1] for row in data]

    # Cria a árvore de decisão
    clf = DecisionTreeClassifier(criterion='gini')

    # Treina a árvore de decisão
    clf.fit(labeled_features, labeled_target)

    # Faz a predição dos dados sem label
    predicted_target = clf.predict(unlabeled_data)

    # Imprime os resultados
    print(predicted_target[:300])

    with open("classificação_gravidade_teste_cego.txt", "w") as f:
        # Iterar sobre os valores e escrevê-los no arquivo
        for value in predicted_target:
            value = int(value)
            f.write(str(value) + "\n") 

    # Plota a árvore
    plot_tree(clf)
    plt.show()