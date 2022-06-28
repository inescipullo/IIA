from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn import tree
import graphviz

import matplotlib.pyplot as plt


# Se carga el conjunto de datos y se divide en conjunto de datos de entrenamiento y conjunto de datos de evaluación en una proporsión de 80/20 
dataset = load_digits(n_class=10, return_X_y=False, as_frame=False)
data_train, data_test, target_train, target_test = train_test_split(dataset.data,
                                                                    dataset.target,
                                                                    test_size = 0.2)

# EJERCICIO 1
# Para graficar muestra de la imagenes:
# plt.matshow(dataset.images[894])
# plt.show()


def default_trainig():
    '''
    Los valores por defecto de los parámetros de DecisionTreeClassifier son:
    criterion: ”gini”
    splitter: ”best”
    max_depth: None
    min_samples_split: 2
    min_samples_leaf: 1
    min_weight_fraction_leaf: 0.0
    max_features: None
    random_state: None
    max_leaf_nodes: None
    min_impurity_decrease: 0.0
    class_weightdict: None
    ccp_alpha: 0.0
    '''
    # Se crea el modelo del Árbol de Decisión con los parámetros por defecto y se entrena el modelo con el conjunto de entrenamiento
    tree_model = DecisionTreeClassifier()
    tree_model.fit(data_train, target_train)

    # Predicción de la accuracy del modelo probandolo con el conjunto de evaluación
    test_target_predicted = tree_model.predict(data_test)
    # Valor de la accuracy del modelo
    test_accuracy = accuracy_score(target_test, test_target_predicted)

    # Se evalua el sobreentrenamiento midiendo el accuracy del modelo sobre el conjunto de entrenamiento
    train_target_predicted = tree_model.predict(data_train)
    train_accuracy = accuracy_score(target_train, train_target_predicted)

    return tree_model, test_accuracy, train_accuracy


# Función para generar el gráfico del árbol
def graph_tree(tree_model, feature_names=None, class_names=None):
    dot_data = tree.export_graphviz(tree_model, out_file=None,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)  
    graph = graphviz.Source(dot_data)
    return graph


# EJERCICIO 2 a
def ejercicio2a():
    # armo un modelo con los parametros por defecto, lo entreno, imprimo los valores de accuracy y grafico el arbol
    tree_model, test_accuracy, train_accuracy = default_trainig()
    print("Entrenamiento con los parámetros por defecto.")
    print(f"Valor de accuracy sobre el conjunto de evaluación: {test_accuracy}.")
    print(f"Valor de accuracy sobre el conjunto de entrenamiento: {train_accuracy}.")

    target_names = []
    for i in range(len(dataset.target_names)):
        target_names += str(dataset.target_names[i])
    graph = graph_tree(tree_model, dataset.feature_names, target_names)
    return graph

ejercicio2a()


# EJERCICIO 2 b

# Los siguientes son considerados parámetros de parada del modelo: 
# max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes

def ejercicio2b():
    max_test_accuracy,max_train_accuracy,min_test_accuracy,min_train_accuracy = (0,0,1,1)
    best_values = (0,0,0,0)
    worst_values = (0,0,0,0)
    best_cases = []
    for max_depth in range(6,17,2):
        for min_samples_split in range(2,11,2):
            for min_samples_leaf in range(1,10,2):
                for max_leaf_nodes in range(70,151,20):
                    tree_model = DecisionTreeClassifier(max_depth = max_depth,
                                                        min_samples_leaf = min_samples_leaf,
                                                        min_samples_split = min_samples_split,
                                                        max_leaf_nodes = max_leaf_nodes)
                    tree_model.fit(data_train, target_train)

                    # Predicción de la accuracy del modelo probandolo con el conjunto de evaluación
                    test_target_predicted = tree_model.predict(data_test)
                    # Valor de la accuracy del modelo
                    test_accuracy = accuracy_score(target_test, test_target_predicted)

                    # Se evalua el sobreentrenamiento midiendo el accuracy del modelo sobre el conjunto de entrenamiento
                    train_target_predicted = tree_model.predict(data_train)
                    train_accuracy = accuracy_score(target_train, train_target_predicted)

                    print(f"max_depth: {max_depth}, min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, max_leaf_nodes: {max_leaf_nodes}")
                    print(f"Valor de accuracy sobre el conjunto de evaluación: {test_accuracy}.")
                    print(f"Valor de accuracy sobre el conjunto de entrenamiento: {train_accuracy}.\n")


                    if min_test_accuracy > test_accuracy and min_train_accuracy > train_accuracy:
                        min_test_accuracy = test_accuracy
                        min_train_accuracy = train_accuracy
                    if max_test_accuracy < test_accuracy and max_train_accuracy < train_accuracy:
                        max_test_accuracy = test_accuracy
                        max_train_accuracy = train_accuracy


                    # Ejercicio 2 c
                    # Busca los modelos cuya accuracy sobre los datos de validación sea mayor al 80% pero que no este sobreentrenado, 
                    # es decir, que la diferencia entre la accuracy sobre entrenamiento y sobre evaluación sea menor al 5%.
                    if ((test_accuracy >= 0.8) and (train_accuracy - test_accuracy <= 0.05)):
                        best_cases.append((train_accuracy, test_accuracy, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes))

    print(f"max_test_accuracy: {max_test_accuracy}, max_train_accuracy: {max_train_accuracy}, min_test_accuracy: {min_test_accuracy}, min_train_accuracy: {min_train_accuracy}\n")


    # Imprimo los modelos selecionados.
    print("Mejores modelos:")
    for tuple in best_cases:
        print(f"train_accuracy: {tuple[0]}, test_accuracy: {tuple[1]}")
        print(f"max_depth: {tuple[2]}, min_samples_split: {tuple[3]}, min_samples_leaf: {tuple[4]}, max_leaf_nodes: {tuple[5]}")

    return 0

# ejercicio2b()


