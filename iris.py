import csv
import math
import time
import random
import matplotlib.pyplot as plt

# Este é um programa simples de IA que lê dados de "iris.data" sobre três tipos de plantas de íris e aprende a
# classificar o tipo de uma planta de íris com base em suas medições fornecidas como entrada. O objetivo disso é
# demonstrar como é simples implementar uma rede neural do zero, sem utilizar bibliotecas específicas, mesmo em uma
# linguagem como python. O tipo de rede neural sendo usada é uma rede feed-forward, usando um algoritmo de descida
# estocástica de gradiente como método de aprendizado.

# Definições de colunas no arquivo iris.data
SEPAL_LENGTH_COLUMN = 0
SEPAL_WIDTH_COLUMN = 1
PETAL_LENGTH_COLUMN = 2
PETAL_WIDTH_COLUMN = 3
IRIS_CLASS_COLUMN = 4

# Variáveis globais
min_sepal_length = float('inf')
max_sepal_length = float('-inf')

min_sepal_width = float('inf')
max_sepal_width = float('-inf')

min_petal_length = float('inf')
max_petal_length = float('-inf')

min_petal_width = float('inf')
max_petal_width = float('-inf')


# Definições de funções
def main():
    # network_structure:
    #   quantidade de elementos = número de camadas
    #   valor do elemento = número de neurônios nessa camada
    #
    # O primeiro e o último elementos são as camadas de entrada e saída, respectivamente. Mude com cuidado.

    network_structure = [4, 5, 3]

    # reps:
    #   número de repetições / iterações de treinamento (epochs).
    reps = 5000
  
    # learning_rate:
    #   É a rapidez com que a rede neural aprende, porém também é uma faca de dois gumes,
    #   pois também controla a rapidez com que a rede neural esquece.
    learning_rate = 0.2

    # Define quantas vezes irá ser executado uma validação, usando um elemento aleatório de iris.data
    qtd_vezes_validacao = 100

    # Coloca um seed aleatório
    random.seed(time.time())

    # Primeiro vamos carregar os dados, os dados são um arquivo CSV simples
    raw_dataset = load_data('iris.data')

    # Em seguida, vamos normalizar para que nossa IA possa entendê-lo.
    #
    # A normalização envolve medir o intervalo de uma determinada amostra de dados e restringir esse intervalo para
    #   que todos os pontos de dados fiquem entre dois valores. Nesta rede neural específica, restringiremos todos os
    #   intervalos de dados entre 0 e 1.
    normalized_dataset = normalize_dataset(raw_dataset)

    # Agora, para construir a rede neural. Como dito acima, a lista network_structure define
    #   a estrutura da rede. Por exemplo, uma lista como [3, 2, 1] criaria uma rede como esta:
    #
    #               o >
    #                       o -\
    #               o >             o
    #                       o -/
    #               o >
    #
    # A arte ASCII é difícil, mas a lista [3, 2, 1] denota uma rede com 3 neurônios de entrada, 1 camada de 2
    # neurônios de camada oculta e 1 neurônio de saída.
    #
    # Da mesma forma, uma estrutura [3, 2, 4, 1] denotaria a mesma coisa acima, mas com duas camadas ocultas,
    # sendo a primeira composta por dois neurônios e a segunda composta por quatro neurônios.
    #
    # Os neurônios neste código serão representados por uma matriz bidimensional de floats, indicando sua saída.
    #
    # Também devemos observar as linhas traçadas nessa figura, essas são as sinapses entre os neurônios. Assim como
    # nosso cérebro, as redes neurais artificiais também têm sinapses. Em uma rede de alimentação direta,
    # cada neurônio em uma camada está totalmente conectado a cada neurônio na próxima camada. Essas sinapses
    # transportam a saída de cada neurônio como um sinal para o neurônio na outra extremidade. As sinapses neste
    # código serão representadas como uma matriz tridimensional de floats, denotando o peso ou a FORÇA de uma
    # sinapse.
    #
    # Por fim, existem as viés (bias) dos neurônios, que geralmente são representados como um neurônio especial
    # por camada que se conecta a todos os neurônios em sua camada com suas próprias sinapses. Esse neurônio de
    # viés/bias gera continuamente o valor 1.0. Tudo isso é inútil para simular, em vez disso, apenas imitaremos a
    # estrutura float dos neurônios, cada valor representando o peso desse viés. Embora esse neurônio não tenha
    # base na realidade, é importante para o aprendizado, pois ajuda a rede a se tornar mais "flexível" na modelagem
    # de um determinado conjunto de dados.
    neurons = create_neurons(network_structure)
    biases = create_biases(network_structure)
    synapses = create_synapses(network_structure)

    # Variáveis para os gráficos
    losses = {'training': [], 'validation': [], 'acertos': []}
    average_loss = {'training': [], 'validation': [], 'acertos': []}

    # Agora estamos prontos para começar a aprender. Observe as funções feedforward, backprop e learn
    for i in range(reps):
        testing_example = random.randint(0, len(normalized_dataset) - 1)
        feed_forward(normalized_dataset[testing_example], neurons, biases, synapses)
        expected_output = []

        if raw_dataset[testing_example][IRIS_CLASS_COLUMN] == 'Iris-setosa':
            # Se for Iris-setosa, a saída esperada seria 1.0 no primeiro elemento da array, e 0 nas outras.
            expected_output = [1.0, 0.0, 0.0]
        elif raw_dataset[testing_example][IRIS_CLASS_COLUMN] == 'Iris-versicolor':
            # Se for Iris-versicolor, a saída esperada seria 1.0 no segundo elemento da array, e 0 nas outras.
            expected_output = [0.0, 1.0, 0.0]
            # Se for Iris-virginica, a saída esperada seria 1.0 no terceiro elemento da array, e 0 nas outras.
        elif raw_dataset[testing_example][IRIS_CLASS_COLUMN] == 'Iris-virginica':
            expected_output = [0.0, 0.0, 1.0]

        training_loss = mse(expected_output, neurons[-1])
        losses['training'].append(training_loss)
        average_loss['training'].append(sum(losses['training']) / len(losses['training']))

        if i & 100 == 0:
            print("Época %d, saída do neurônio %s" % (i, neurons[-1]))

        errors = backprop(expected_output, neurons, synapses)
        learn(learning_rate, errors, neurons, biases, synapses)

    # Fase de validação, é como uma espécie de exame final para nossa rede.
    #   Desativamos o aprendizado e apenas o deixamos executar.
    for i in range(qtd_vezes_validacao):
        testing_example = random.randint(0, len(normalized_dataset) - 1)
        feed_forward(normalized_dataset[testing_example], neurons, biases, synapses)
        expected_output = []

        if raw_dataset[testing_example][IRIS_CLASS_COLUMN] == 'Iris-setosa':
            expected_output = [1.0, 0.0, 0.0]
        elif raw_dataset[testing_example][IRIS_CLASS_COLUMN] == 'Iris-versicolor':
            expected_output = [0.0, 1.0, 0.0]
        elif raw_dataset[testing_example][IRIS_CLASS_COLUMN] == 'Iris-virginica':
            expected_output = [0.0, 0.0, 1.0]

        validation_loss = mse(expected_output, neurons[-1])
        losses['validation'].append(validation_loss)
        average_loss['validation'].append(sum(losses['validation']) / len(losses['validation']))

        planta_mais_provavel_index = neurons[-1].index(max(neurons[-1]))
        planta_mais_provavel = ""

        if planta_mais_provavel_index == 0:
            planta_mais_provavel = 'Iris-setosa'
        elif planta_mais_provavel_index == 1:
            planta_mais_provavel = 'Iris-versicolor'
        elif planta_mais_provavel_index == 2:
            planta_mais_provavel = 'Iris-virginica'

        acertou = raw_dataset[testing_example][IRIS_CLASS_COLUMN] == planta_mais_provavel

        print("Validação: %d - Saída do neurônio: %s" % (i, neurons[-1]))
        print("Validação: %d - A planta correta: é %s, e o algoritmo adivinha que a planta mais provável de ser é: %s"
              % (i, raw_dataset[testing_example][IRIS_CLASS_COLUMN], planta_mais_provavel))
        print("Validação: %d - O algoritmo acertou? %s"
              % (i, acertou))
        print()

        losses['acertos'].append(acertou)
        average_loss['acertos'].append(planta_mais_provavel)

    print()
    print("Quantidade de acertos com dados de validação: %d" % (losses['acertos'].count(True)))
    print("Quantidade de erros com dados de validação: %d" % (losses['acertos'].count(False)))
    print()
    print("O algoritmo gosta de adivinhar mais a seguinte planta: %s"
          % (max(set(average_loss['acertos']), key=average_loss['acertos'].count)))


    # Plote os resultados em bons gráficos
    plt.plot(average_loss['training'], label='Perda de treinamento')
    plt.legend()
    plt.ylim(ymax=1)
    plt.xlabel('Epoca')
    plt.ylabel('Taxa de erro')
    plt.savefig("perda_treinamento.png")

    # Limpa o gráficos
    plt.clf()
    
    # Finalizado!


def load_data(filename):
    global min_sepal_length
    global max_sepal_length
    global min_sepal_width
    global max_sepal_width
    global min_petal_length
    global max_petal_length
    global min_petal_width
    global max_petal_width
    global SEPAL_LENGTH_COLUMN
    global SEPAL_WIDTH_COLUMN
    global PETAL_LENGTH_COLUMN
    global PETAL_WIDTH_COLUMN
    global IRIS_CLASS_COLUMN

    count = 0
    dataset = []

    with open(filename) as data_file:

        csv_reader = csv.reader(data_file, delimiter=',')

        for row in csv_reader:
            if not row:
                continue

            dataset.append([
                float(row[0]), float(row[1]), float(row[2]), float(row[3]), row[4]
            ])

            min_sepal_length = min(dataset[count][SEPAL_LENGTH_COLUMN], min_sepal_length)
            max_sepal_length = max(dataset[count][SEPAL_LENGTH_COLUMN], max_sepal_length)

            min_sepal_width = min(dataset[count][SEPAL_WIDTH_COLUMN], min_sepal_width)
            max_sepal_width = max(dataset[count][SEPAL_WIDTH_COLUMN], max_sepal_width)

            min_petal_length = min(dataset[count][PETAL_LENGTH_COLUMN], min_petal_length)
            max_petal_length = max(dataset[count][PETAL_LENGTH_COLUMN], max_petal_length)

            min_petal_width = min(dataset[count][PETAL_WIDTH_COLUMN], min_petal_width)
            max_petal_width = max(dataset[count][PETAL_WIDTH_COLUMN], max_petal_width)

            count += 1
    return dataset


# Precisamos normalizar os valores para que todos eles estejam de uma forma que as redes neurais possam entender
# melhor. Como estamos usando a função de ativação sigmóide, ela retorna valores entre 0 e 1, o que significa que
# todas as entradas precisam estar no mesmo formato.
#
# Podemos fazer isso usando esta fórmula que usa os valores mínimo e máximo da coluna:
#   z_i = x_i - min_x / x_max - min_x, em que z_i é o valor normalizado entre 0 e 1, com min_x o menor valor
#   na coluna e max_x sendo o maior.
#
# Quanto ao IRIS_CLASS, podemos realmente descartar esses valores para o conjunto de dados normalizados. O objetivo
#   deste projeto é classificar qual tipo de íris é uma determinada medida de pétala e sépala; a IA descobrirá isso
#   sozinha
def normalize(x, min_x, max_x):
    return (x - min_x) / (max_x - min_x)


def normalize_dataset(dataset):
    global min_sepal_length
    global max_sepal_length
    global min_sepal_width
    global max_sepal_width
    global min_petal_length
    global max_petal_length
    global min_petal_width
    global max_petal_width
    global SEPAL_LENGTH_COLUMN
    global SEPAL_WIDTH_COLUMN
    global PETAL_LENGTH_COLUMN
    global PETAL_WIDTH_COLUMN
    global IRIS_CLASS_COLUMN

    normalized_dataset = []

    for row in dataset:
        normalized_sepal_length = normalize(row[SEPAL_LENGTH_COLUMN], min_sepal_length, max_sepal_length)
        normalized_sepal_width = normalize(row[SEPAL_WIDTH_COLUMN], min_sepal_width, max_sepal_width)
        normalized_petal_length = normalize(row[PETAL_LENGTH_COLUMN], min_petal_length, max_petal_length)
        normalized_petal_width = normalize(row[PETAL_WIDTH_COLUMN], min_petal_width, max_petal_width)

        normalized_dataset.append([
            normalized_sepal_length,
            normalized_sepal_width,
            normalized_petal_length,
            normalized_petal_width
        ])

    return normalized_dataset


def create_neurons(network_structure):
    return [[0] * neurons for neurons in network_structure]


def create_biases(network_structure):
    return [[random.random() for _ in range(neurons)] for neurons in network_structure]


def create_synapses(network_structure):
    return [
        [[random.random() for post_synaptic_neurons in range(network_structure[layer + 1])]
         for pre_synaptic_neurons in range(network_structure[layer])]
        for layer in range(len(network_structure) - 1)]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def d_sigmoid(x):
    return x * (1.0 - x)


# O erro quadrático médio é uma maneira simples de calcular o erro, ou seja, é uma função de custo
def mse(expected_output, actual_output):
    return sum([(expected_output[i] - actual_output[i]) ** 2 for i in range(len(actual_output))]) / 2


# Alimentar entradas em uma rede neural consiste apenas do seguinte:
#    1. Atribuir o valor da entrada na saída dos neurônios da camada de entrada. Esses neurônios não fazem
#        alguma coisa interessante.
#
#    2. Itere sobre cada neurônio na próxima camada, cada neurônio somará a saída de cada neurônio antes dele,
#        multiplicado pelo peso sináptico que os conecta. Em seguida, ele adicionará o valor do peso da polarização
#        ao resultado.
#
#        Por fim, ele pega essa soma e aplica uma função de ativação, ou seja:
#
#            função_de_ativação(somatório(saída * peso) + viés).
#
#        Uma função de ativação replica o comportamento de pico dos neurônios da vida real com base na entrada
#        que recebe.
#
#        É representado aqui com uma função de ativação sigmóide. Esse valor agora é o valor de saída do neurônio.
#
#    3. O passo acima é repetido por toda a camada oculta e pela camada de saída.
def feed_forward(inputs, neurons, biases, synapses):
    for neuron in range(len(neurons[0])):
        neurons[0][neuron] = inputs[neuron]
    for layer in range(1, len(neurons)):
        for post_synaptic_neuron in range(len(neurons[layer])):
            summation = biases[layer][post_synaptic_neuron]
            for pre_synaptic_neuron in range(len(neurons[layer - 1])):
                summation += neurons[layer - 1][pre_synaptic_neuron] \
                             * synapses[layer - 1][pre_synaptic_neuron][post_synaptic_neuron]
            neurons[layer][post_synaptic_neuron] = sigmoid(summation)


# Para encurtar a história, calculamos o erro na saída da rede. Usando esse erro, podemos determinar o erro dos
# neurônios acima da saída, camada por camada, até chegarmos ao topo. Em outras palavras, estamos otimizando uma
# função de custo para encontrar os parâmetros específicos para fornecer o menor custo, ou seja, menor perda,
# ou seja, menor erro.
def backprop(expected, neurons, synapses):
    errors = [[0] * len(neurons[layer]) for layer in range(len(neurons))]
    for neuron in range(len(neurons[-1])):
        errors[-1][neuron] = (expected[neuron] - neurons[-1][neuron]) * d_sigmoid(neurons[-1][neuron])
    for layer in reversed(range(len(neurons) - 1)):
        for pre_synaptic_neuron in range(len(neurons[layer])):
            error = 0
            for post_synaptic_neuron in range(len(neurons[layer + 1])):
                error += errors[layer + 1][post_synaptic_neuron] \
                         * synapses[layer][pre_synaptic_neuron][post_synaptic_neuron]
            errors[layer][pre_synaptic_neuron] = error * d_sigmoid(neurons[layer][pre_synaptic_neuron])
    return errors


# Isso está colocando os erros em prática para perturbar os pesos e obter os valores que queremos da rede neural
def learn(learning_rate, errors, neurons, biases, synapses):
    for layer in range(1, len(neurons)):
        for pre_synaptic_neuron in range(len(neurons[layer - 1])):
            for post_synaptic_neuron in range(len(neurons[layer])):
                synapses[layer - 1][pre_synaptic_neuron][post_synaptic_neuron] \
                    += learning_rate * errors[layer][post_synaptic_neuron] * neurons[layer - 1][pre_synaptic_neuron]
    for layer in range(1, len(neurons)):
        for neuron in range(len(neurons[layer])):
            biases[layer][neuron] += learning_rate * errors[layer][neuron] * 1.0


main()
