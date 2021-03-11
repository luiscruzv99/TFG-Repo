import Modelo.model as md
import Modelo.training as tn
import pickle as pk
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import time as tm


'''
Funcion que detecta que dispositivo puede usar pytorch para entrenar la red
'''
def check_device():

    if torch.cuda.is_available():
        device = torch.device('cuda')  # Entrenar con la GPU primaria
    else:
        device = torch.device('cpu')  # Entrenar con el CPU

    return device

'''
Funcion que carga en arrays los datos de los archivos pickle
'''
def load_data():
    
    with open('Entrenamiento/data.pkl','rb') as f:
        data = pk.load(f)
    with open('Entrenamiento/labels.pkl','rb') as f:
        labels = pk.load(f)

    return [data, labels]


'''
Funcion que convierte de los arrays en tensores de pytorch, mandandolos a la GPU, definicion de datasets y dataloaders
'''
def convert_data(data, labels, device):
    
    # Definicion de los tensores y dataloaders de la fase de entrenamiento
    train_data = torch.Tensor(data[:4684]).to(device) 
    train_labels = torch.Tensor(labels[:4684]).to(device) 

    train_set = TensorDataset(train_data,train_labels)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)

    # Definicion de los tensores y dataloaders de la fase de validacion 
    val_data = torch.Tensor(data[30000:30750]).to(device)
    val_labels = torch.Tensor(labels[30000:30750]).to(device)

    val_set = TensorDataset(val_data, val_labels)
    val_loader = DataLoader(val_set, shuffle=True)

    return [train_loader, val_loader]



device = check_device()
print("Usando dispositivo " + str(device))

st = tm.time()
data, labels = load_data()
print("Cargado de los datos: "+str(tm.time()-st))

st = tm.time()
train_loader, val_loader = convert_data(data=data, labels=labels, device=device)
print("Creacion de datasets: "+str(tm.time()-st))

# Definicion de la red neuronal, optimizador y perdida
net = md.resNet34(in_channels=1,n_classes=2).to(device)  # Instancia de ResNet34
loss = torch.nn.MSELoss(reduction='mean').to(device)  # Funcion de perdida 
optimizador = torch.optim.SGD(net.parameters(), lr=0.128)  # Optimizador de la red (Stochastic Gradient Deescent)

# Fase de entrenamiento 
st = tm.time()
tn.train(train_loader, net, optimizador)
print("Tiempo de entrenamiento: "+str(tm.time()-st))

# Fase de validacion
st = tm.time()
acc, loss = tn.validate_or_test(val_loader, net, optimizador)
print("Tiempo de validacion: "+str(tm.time()-st))
print("Precision: "+str(acc))
print("Perdida: "+str(loss))
