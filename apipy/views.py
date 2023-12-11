from django.http import HttpResponse
import numpy as np
import pandas as pd
from sklearn import tree
import joblib
import os

def train(request):
    file_path = os.path.join(os.path.expanduser("~"), "SampleDjango", "tic_tac_toe.csv")
    training = pd.read_csv(file_path)
    #training = pd.read_csv("apipy/tic_tac_toe.csv")
    training["top-left-square"] = training["top-left-square"].apply(lambda toLabel: 0 if toLabel == 'x'  else 1 if toLabel == 'o'  else 2)
    training["top-middle-square"] = training["top-middle-square"].apply(lambda toLabel: 0 if toLabel == 'x'  else 1 if toLabel == 'o'  else 2)
    training["top-right-square"] = training["top-right-square"].apply(lambda toLabel: 0 if toLabel == 'x'  else 1 if toLabel == 'o'  else 2)
    training["middle-left-square"] = training["middle-left-square"].apply(lambda toLabel: 0 if toLabel == 'x'  else 1 if toLabel == 'o'  else 2)
    training["middle-middle-square"] = training["middle-middle-square"].apply(lambda toLabel: 0 if toLabel == 'x'  else 1 if toLabel == 'o'  else 2)
    training["middle-right-square"] = training["middle-right-square"].apply(lambda toLabel: 0 if toLabel == 'x'  else 1 if toLabel == 'o'  else 2)
    training["bottom-left-square"] = training["bottom-left-square"].apply(lambda toLabel: 0 if toLabel == 'x'  else 1 if toLabel == 'o'  else 2)
    training["bottom-middle-square"] = training["bottom-middle-square"].apply(lambda toLabel: 0 if toLabel == 'x'  else 1 if toLabel == 'o'  else 2)
    training["bottom-right-square"] = training["bottom-right-square"].apply(lambda toLabel: 0 if toLabel == 'x'  else 1 if toLabel == 'o'  else 2)
    training["Class"] = training["Class"].apply(lambda toLabel: 0 if toLabel == 'positive' else 1)

    columns = ["Class","top-left-square", "top-middle-square", "top-right-square", "middle-left-square", "middle-middle-square","middle-right-square","bottom-left-square","bottom-middle-square","bottom-right-square"]
    #create the variable to hold the features that the classifier will use
    #datosiniciales = training[list(columns)].values
    datosiniciales = training[list(columns)]

    #Datos de entrada
    columns = ["top-left-square", "top-middle-square", "top-right-square", "middle-left-square", "middle-middle-square","middle-right-square","bottom-left-square","bottom-middle-square","bottom-right-square"]
    X_input = datosiniciales[list(columns)].values

    #Target el objetivo o salida deseada
    y_target = datosiniciales["Class"].values

    #create clf_train as a decision tree classifier object
    clf_train = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)

    #train the model using the fit() method of the decision tree object. 
    #Supply the method with the input variable X_input and the target variable y_target
    clf_train = clf_train.fit(X_input, y_target)

    file_path2 = os.path.join(os.path.expanduser("~"), "SampleDjango", "decision_tree_model.joblib")
    # Save the model to a file
    joblib.dump(clf_train, file_path2)
    
    return HttpResponse("Training successful")

def legal_moves_generator(current_board_state,turn_monitor):
    legal_moves_dict={}
    for i in range(current_board_state.shape[0]):
        for j in range(current_board_state.shape[1]):
            if current_board_state[i,j]==2:
                board_state_copy=current_board_state.copy()
                board_state_copy[i,j]=turn_monitor
                legal_moves_dict[(i,j)]=board_state_copy.flatten()
    return legal_moves_dict

def fetch_next_move(legal_moves_dict):
  file_path = os.path.join(os.path.expanduser("~"), "SampleDjango", "decision_tree_model.joblib")
  clf_train = joblib.load(file_path)
  
  for i,j in legal_moves_dict:
    if clf_train.predict([legal_moves_dict[(i,j)]]) == 0:
      return np.array(legal_moves_dict[(i,j)])
  for i,j in legal_moves_dict:
      return np.array(legal_moves_dict[(i,j)])
    
def find_different_element(array1, array2):
    # Encuentra los índices donde los elementos son diferentes
    different_index = np.where(array1 != array2)[0]

    # Si hay elementos diferentes, devuelve el primer índice encontrado
    if len(different_index) > 0:
        return different_index[0]
    else:
        # Si no hay elementos diferentes, devuelve -1
        return -1
  
def recibe(request, x0, x1, x2, x3, x4, x5, x6, x7, x8, turn_monitor):
    # Crear el dataframe
    current_board_state = np.array([[x0, x1, x2], [x3, x4, x5], [x6, x7, x8]])

    current_board_state = current_board_state.astype(int)

    # Obtener los movimientos legales
    legal_moves_dict = legal_moves_generator(current_board_state, turn_monitor)

    print("Diccionario:", legal_moves_dict)

    next_move = fetch_next_move(legal_moves_dict)
    print("\nindice next:",next_move)
    current_board_state = current_board_state.reshape(9)

    indice_diferente = find_different_element(next_move,current_board_state)

    # Return the response as json
    return HttpResponse(indice_diferente)