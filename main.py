import sys
import os

# Obtém o diretório raiz do projeto
root_dir = os.path.dirname(os.path.abspath(__file__))
# Adiciona o diretório raiz ao sys.path
sys.path.append(root_dir)

from sort.tree_sort import tree_sort
from regression.neural_regression import neural_regressor
from sort.fuzzy_gaussian import fuzzy_gaussian


opt = -1
while opt != 0:
    print("\nEscolha o algoritmo\n1 - Tree sort\n2 - Fuzzy\n3 - Neural regresion\n0 - Sair")
    try:
        opt = int(input())
    except ValueError:
        print("\nFavor digitar um número")
        continue
    if(opt == 1):
        tree_sort()
    elif(opt == 2):
        fuzzy_gaussian()
    elif(opt == 3):
        neural_regressor()
    elif(opt == 0):
        print("\nFinalizado!")
    else:
        print("\nOpção inválida")