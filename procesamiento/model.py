from os import name
from mip.constants import MAX
from generador_instancia import GeneradorInstancia
from rating_function import process_ratings, global_register
from parameters import *
import numpy as np
import pandas as pd
import pickle
from mip import Model, xsum, MAXIMIZE, BINARY, CONTINUOUS, CBC, maximize

### CARGAR DATOS ###
file = open('servicios.pickle', 'rb')
services = pickle.load(file)
file.close()

file = open('padres.pickle', 'rb')
parents = pickle.load(file)
file.close()

file = open('casos.pickle', 'rb')
cases = pickle.load(file)
file.close()

file = open('abogados.pickle', 'rb')
lawyers = pickle.load(file)
file.close()

file = open('decodificacion.pickle', 'rb')
specialties_decod = pickle.load(file)
file.close()

file = open('decod_nombres.pickle', 'rb')
lawyers_decod = pickle.load(file)
file.close()

### SELECCIONAR CASO A SOLUCIONAR ###

"""print(f"Seleccionar el caso a resolver: (numero entre el 0 al {len(cases) - 1})\n")
while True:
    case_idx = input("Ingrese un numero: ")
    if case_idx.isdigit():
        if int(case_idx) < len(cases):
            break
        else:
            print(f"El numero debe estar entre 0 y {len(cases) - 1}")
    else:
        print("Se debe ingresar un numero")
case_idx = int(case_idx)
case = cases[case_idx]
print("Los servicios de este caso son:")
for service in case:
    print(f"   - {specialties_decod[service]}.")"""


### GENERAR CASOS FUTUROS ###
instance = GeneradorInstancia(cases, services, lawyers, parents)
service_classes, active, table = instance.inicializar_generador()

print(active)


m = Model(name="Asignacion de abogados", sense=MAXIMIZE, solver_name=CBC)

### PREPROCESAR DATOS ###
#register = global_register(services, parents, lawyers)
#r = process_ratings(lawyers, case, parents, register)


