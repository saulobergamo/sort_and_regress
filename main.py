from sort.tree_sort import tree_sort


opt = -1
while opt != 0:
    print("\nEscolha o algoritmo\n1 - Tree sort\n2 - Fuzzy\n3 - Neural classifier\n0 - Sair")
    try:
        opt = int(input())
    except ValueError:
        print("\nFavor digitar um número")
        continue
    if(opt == 1):
        tree_sort()
    elif(opt == 0):
        print("\nFinalizado!")
    else:
        print("\nOpção inválida")