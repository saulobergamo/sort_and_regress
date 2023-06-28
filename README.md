# sort_and_regress
Universidade Tecnológica Federal do Paraná  - UTFPR -  Sistemas Inteligentes (CSI30)

<font color="red">IMPORTANTE!</font>
- Dependências:

    > python versão 3.8<br>
    matplotlib<br>
    sklearn<br>
    csv<br>
    pandas<br>
    skfuzzy<br>

<font color="red">Execução</font>

    Tenha python3 e pip install.
    pip install <dependência pendente>
    python3 main.py


___
<br>


## <font color="darkblue">Tarefa 2: Classificaçao e Regressao</font>
<br> Resgate de Vítimas de Catástrofes Naturais, Desastres ou Grandes Acidentes


### <font color="blue">1 Objetivo da tarefa</font>
<br>
Nesta tarefa, você tem disponível um histórico de sinais que foram coletados de outros acidentes e analisados por um corpo médico:
<br><br>
• qPA: qualidade da pressão arterial; resulta da avaliação da relação entre a pressão sistólica e a diastólica;

• pulso: pulsação ou Batimento por Minuto (pulso);

• frequência respiratória: frequência da respiração por minuto;

• gravidade: um valor calculado em função dos sinais vitais acima;

• classes de gravidade: são 4 classes que apresentam o estado de saúde do acidentado.

O corpo médico construiu uma fórmula para calcular a gravidade do estado de saúde das vítimas e, também, estabeleceram intervalos  que definem as seguintes classes de gravidade:

>1 = crítico,

>2 = instável,

>3 = potencialmente estável e

>4 = estável.

O problema é que a fórmula de cálculo e os intervalos de gravidade foram perdidos. Portanto, você deve utilizar técnicas de aprendizado de máquina para reconstituir a classificação nas
quatro categorias e o cálculo do valor da gravidade (regressão).
Com base nos modelos aprendidos, o desempenho final será calculado com dados de teste que não foram utilizados no treinamento e validação. Os dados de teste serão passados em
sala de aula no dia da entrega.

Portanto, a tarefa tem dois objetivos:
1) comparar os resultados produzidos por duas técnicas diferentes de classificação, Árvores Indutivas e Fuzzy, dentre as vistas no curso, capazes de realizarem classificação.<br><br>
2) Realizar regressão utilizando Redes Neurais.

A comparação deve ser feita utilizando-se as métricas adequadas ao tipo da tarefa (classificação ou regressão) durante as fases de treinamento/validação e testes (RMSE, precisão, recall, f-measure, acurácia, matriz de confusão) 

### <font color="blue">2 Arquivos de treinamento/validação e testes</font>

#### <font color="lightblue">2.1 Arquivo sinaisvitais_hist.txt</font>

Este arquivo contém os dados históricos de sinais vitais de vítimas de outros acidentes. Cada linha representa uma vítima.

Para uma vítima i do histórico temos 5 sinais vitais (s1 até s5) que resultam a gravidade gi da vítima. Todos os valores são números reais criados de modo randômico dentro dos intervalos apresentados.

| 𝑖 | si1 | si2 | si3 | si4 | si5 | gi | 𝑦𝑖 |
|--|-----|-----|-----|-------|---|----|---|

𝑖: identificação da vítima (número sequencial)

𝑠𝑖1: pressão sistólica (pSist): [5, 22] - não usar, é utilizada no cálculo de 𝑠𝑖3

𝑠𝑖2: pressão diastólica (pDiast): [0, 15] - não usar, é utilizada no cálculo de 𝑠𝑖3

𝑠𝑖3: qualidade da pressão (qPA): [-10,10] onde 0 é a qualidade máxima -10 é a pior qualidade quando a pressão está excessivamente baixa, +10 é a pior qualidade quando a pressão
está excessivamente alta

𝑠𝑖4: pulso: [0,200] bpM

𝑠𝑖5: respiração: [0,22] FpM (frequência de respiração)

𝑔𝑖: gravidade: deve ser inferido pela técnica escolhida

𝑦𝑖 : rótulo que representa a classe de saída: deve ser inferida com base na gravidade (pós-processamento) ou produzida diretamente pela técnica (e.g. árvore de decisão produz diretamente).

Exemplo:

| i | si1    | si2    | si3     | si4     | si5    | gi        | yi     |
|---|--------|--------|---------|---------|--------|-----------|--------|
| i | pSist  | pDiast | qPA     | pulso   | resp   | gravidade | classe |
| 1 | 8.5806 | 2.2791 | -8.4577 | 56.8384 | 9.2229 | 33.5156   | 2      |

#### <font color="lightblue">2.2 Arquivo sinaisvitais_teste.txt</font>

O dataset para o teste cego segue quase o mesmo formato dos dados históricos. No entanto, retiramos si1, si2, g1 e y1. Este arquivo vai ser utilizado somente na fase de teste cego do modelo aprendido para cada os classificadores (Fuzzy e Árvore) e o regressor (RN) a ser fornecido no dia
da entrega para podermos comparar as soluções dos diferentes grupos.


| i | si3     | si4     | si5    |
|---|---------|---------|--------|
| i | qPA     | pulso   | resp   |
| 1 | -8.5577 | 56.8004 | 9.0000 |


Para cada um dos n exemplos do teste cego, cada um dos classificadores deve gerar um arquivo .txt a parte contendo uma coluna com as classes preditas.

| 2   |
|-----|
| 3   |
| ... |
| 1   |

Para cada exemplo do teste cego, o regressor em Redes Neurais deve gerar um arquivo .txt a parte contendo uma coluna com os valores de gravidade preditos.

| 33.5034 |
|---------|
| 10.4034 |
| ...     |
| 0.0399  |

O professor fornecerá os valores conhecidos de gravidade e da classe e o grupo fará o cálculo de erro (RMSE – Raiz Quadrada do Erro Quadrático Médio1) e de classificação (precision, recall, f-measure, acuracidade) para podermos comparar as soluções.

<img src="https://render.githubusercontent.com/render/math?math=\sqrt[]{\frac{1}{n}\sum_{1}^{n}\left(g'_i-g_i\right)^2}"> , 𝑡𝑎𝑙 𝑞𝑢𝑒 𝑔̂𝑖 é 𝑜 𝑎𝑙𝑣𝑜 𝑒 𝑔𝑖, 𝑜 𝑣𝑎𝑙𝑜𝑟 𝑝𝑟𝑒𝑑𝑖𝑡𝑜

### <font color="blue">3 METODOLOGIA</font>

Podem ser utilizadas Toolbox (e.g. MatLab) ou programação com auxílio de bibliotecas existentes (e.g. Python com SciKit, Tensorflow). O importante é entender conceitualmente os parâmetros a serem definidos/implementados (não utilizar ferramentas de maneira cega – sem entender os conceitos).
Para cada técnica utilizada, treinar e validar modelos com diferentes estruturas e parametrizações. Por exemplo, se você utilizar um sistema de inferência fuzzy (SIF) poderá mudar as variáveis linguísticas que caracterizam as entradas (o total de termos linguísticos, as funções de pertinência que os definem). No caso particular de um SIF, as regras podem ser
construídas manualmente ou você pode implementar o método de Wang-Mendel para gerá-las automaticamente. Neste caso, terá um bônus na nota final.
Ainda, para extrair um comportamento médio independente da escolha dos dados de treinamento/validação, você deve fazer a validação cruzada várias vezes para cada configuração escolhida. A partir daí, você seleciona o modelo que gerou o melhor resultado para utilizá-lo na fase de testes. Esta fase permite analisar a capacidade de generalização do modelo aprendido. Portanto, pode haver casos em que um modelo com bom desempenho na etapa de treinamento/avaliação não seja tão bom na etapa de testes, indicando sobreadaptação
aos dados de treinamento (overfitting).

### <font color="blue">4 ENTREGA</font>
1) Os códigos fonte. Caso utilize uma Toolbox, descrever qual foi utilizada,
parametrização e scripts.
2) Um artigo PDF de até 10 páginas no formato da SBC com a estrutura abaixo

#### <font color="lightblue">4.1 Estrutura do artigo</font>

* Introdução: dentro do problema como um todo, quais subproblemas atacará e por quais razões: quais são as motivações e justificativas para resolvê-los. Fundamentação Teórica: as técnicas escolhidas com uma breve descrição 
* Metodologia: descreva como procedeu para avaliar cada uma das técnicas escolhidas, salientando as variações de parametrização e de estrutura (e.g. regras fuzzy, topologia da rede neural). Explicar a razão de tentar uma nova parametrização e/ou estrutura. Explicar
como procedeu a validação cruzada (quantas execuções, qual critério de seleção do modelo a ser usado nos testes)
* Resultados e análise: mostrar os resultados numéricos das métricas de desempenho para as etapas de treinamento/avaliação e para a etapa de testes. Fazer uma análise comparativa entre as técnicas escolhidas.
* Conclusões: qual técnica apresentou o melhor desempenho e as razões que você
crê que justificam o desempenho. Há algo a ser melhorado nas soluções apresentadas?
* Referências bibliográficas
* Apêndice: instruções claras de como executar o código respeitando os formatos de arquivos de entrada e de configuração do enunciado; print das telas do programa se desejar (não colocar print das telas no corpo do artigo).

### <font color="blue">5 Critérios de correção dos projetos</font>

• Problema: nível de dificuldade

• Fundamentação Teórica: emprego correto dos termos e conceitos

• Abordagens Relacionadas: qualidade e atualidade do levantamento bibliográfico

• Proposta: qualidade, detalhamento e correção da proposta

• Comparação: quais são as abordagens de comparação

• Análise: qualidade da análise dos resultados e método de treinamento/validação e teste

• Geral: apresentação geral e qualidade da redação

• Bônus: Wang-Mendel para sistemas Fuzzy
