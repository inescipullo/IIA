from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


# Se carga el conjunto de datos y se divide en conjunto de datos de entrenamiento y conjunto de datos de evaluación en una proporsión de 80/20 
dataset = load_digits(n_class=10, return_X_y=False, as_frame=False)
data_train, data_test, target_train, target_test = train_test_split(dataset.data,
                                                                    dataset.target,
                                                                    test_size = 0.2)

# Normalización del conjunto de datos
scaler = StandardScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)


# Si no se pasa el parámetro, es el entrenamiento por defecto.
# Si se define el learning rate es el entrenamiento por defecto pero modificando el valor de leraning_rate_init.
def default_training(learning_rate_init=0.001):
    '''
    Parámetros por defecto de la función MLPClassifier (que crea el modelo)
    hidden_layer_sizes: (100,)
    activation: 'relu'
    solver: 'adam'
    alpha: 0.0001
    batch_size: 'auto'
    learning_rate: 'constant'
    learning_rate_init: 0.001
    power_t: 0.5
    max_iter: 200
    shuffle: True
    random_state: None
    tolfloat: 1e-4
    verbose: False
    warm_start: False
    momentum: 0.9
    nesterovs_momentum: True
    early_stopping: False
    validation_fraction: 0.1
    beta_1: 0.9
    beta_2: 0.999
    epsilon: 1e-8
    n_iter_no_change: 10
    max_fun: 15000
    '''
    # Se crea el modelo de la Red Neuronal con los parámetros por defecto y se entrena el modelo con el conjunto de entrenamiento
    mlp_model = MLPClassifier(learning_rate_init = learning_rate_init)
    mlp_model.fit(data_train, target_train)

    # Predicción de la accuracy del modelo probandolo con el conjunto de evaluación
    target_test_predicted = mlp_model.predict(data_test)
    # Valor de la accuracy del modelo
    test_accuracy = accuracy_score(target_test, target_test_predicted)

    # Se evalua el sobreentrenamiento midiendo el accuracy del modelo sobre el conjunto de entrenamiento
    target_train_predicted = mlp_model.predict(data_train)
    train_accuracy = accuracy_score(target_train, target_train_predicted)
    
    print(f"Valor de accuracy sobre el conjunto de evaluación: {test_accuracy}.")
    print(f"Valor de accuracy sobre el conjunto de entrenamiento: {train_accuracy}.")
    
    return mlp_model, test_accuracy, train_accuracy
    
# default_training()

def accuracy_plotting():
    learning_rate_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    test_accuracy_values = []
    train_accuracy_values = []
    for value in learning_rate_values:
        print(f"Learning rate del modelo: {value}")
        _mlp_model, test_accuracy, train_accuracy = default_training(value)
        print()
        test_accuracy_values.append(test_accuracy)
        train_accuracy_values.append(train_accuracy)

    plt.plot(learning_rate_values, train_accuracy_values, 'ro')
    plt.plot(learning_rate_values, test_accuracy_values, 'bo')
    plt.title('Accuracy del modelo según su learning rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()

    return 0

accuracy_plotting()