from os import name
from mip.constants import MAX
from generador_instancia import GeneradorInstancia
from rating_function import process_ratings, global_register
from parameters import *
import numpy as np
import pandas as pd
import pickle
from mip import Model, xsum, MAXIMIZE, BINARY, CONTINUOUS, CBC, maximize, OptimizationStatus
import random

### FIJAR SEMILLA ###
random.seed(10)

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
instance = GeneradorInstancia(cases, services, lawyers, parents, N=20)
service_classes, active, table = instance.inicializar_generador()

P = [i for i in range(1, instance.horizonte + 1)] # conjunto de periodos
E = [i for i in range(1, instance.n_escenarios + 1)] # conjunto de escenarios
L = [id for id in lawyers["id"]] # conjunto de abogados

# active[scenario, period] = [active cases]
case_ids = [s.id for s in service_classes if s.escenario == 0]
s_ids = {s.ngen: s.id for s in service_classes}

S = [s.ngen for s in service_classes]
S0 = [s.ngen for s in service_classes if s.escenario == 0]
S_arrival = {(e, p): [s.ngen for s in service_classes if s.escenario == e and s.periodo == p]
             for e in E for p in P}
register = global_register(services, parents, lawyers)

### PARAMETROS ###

r = process_ratings(lawyers, s_ids.values(), parents, register) #rating
d = lawyers["hb"].to_dict()
h = {s.ngen: s.hsemanas for s in service_classes}
H = {s.ngen: s.semanas for s in service_classes}
M = 10**8

### MODELO ###
print("Creando modelo...")

m = Model(name="Asignacion de abogados", sense=MAXIMIZE, solver_name=CBC)

## VARIABLES ##

x = [[m.add_var(name=f"x_({l}, {s})", var_type=BINARY) for s in S] for l in L]
y = [m.add_var(name=f"y_{s}", var_type=CONTINUOUS) for s in S]
t = [[m.add_var(name=f"t_({l}, {s})", var_type=CONTINUOUS) for s in S] for l in L]
z = {(l, e, p): m.add_var(name=f"z_({l}, {e}, {p})") for p in P for e in E for l in L}
n = [m.add_var(name=f"n_{s}", lb=0) for s in S]

## FUNCION OBJETIVO ##
m.objective = maximize(
    xsum(H[s] * xsum(t[l][s] * r[l, s_ids[s]] for l in L) for s in S0) + 
    1 / len(E) * xsum((LAMBDA ** p) * xsum(H[s] * xsum(t[l][s] * r[l, s_ids[s]] for l in L) 
    for s in S_arrival[e, p]) for e, p in S_arrival.keys())
)

## RESTRICCIONES ##
# R1 #
for s in S:
    m += xsum(t[l][s] for l in L) == h[s] * (1 - n[s]), f"r1_{s}"

    # R4 #
    m += y[s] == xsum(x[l][s] for l in L), f"r4_{s}"

    # R5 #
    m += n[s] >= 1 - y[s], f"r5_{s}"

    # R2 #
    for l in L:
        m += T_MIN * x[l][s] <= t[l][s], f"r2.1_({l}, {s})"
        m += M * x[l][s] >= t[l][s], f"r2.2_({l}, {s})"

        # R3 #
        if r[l, s_ids[s]] == 0:
            m += x[l][s] == 0, f"r3_({l}, {s})"

# R6 #
for l in L:
    for e in E:
        for p in P:
            m += z[l, e, p] == d[l] - xsum(t[l][s] for s in active[e, p]), f"R6_({l}, {e}, {p})"

print("Modelo creado.")

m.max_gap = 0.05
status = m.optimize(max_seconds=300)
if status == OptimizationStatus.OPTIMAL:
    print('optimal solution cost {} found'.format(m.objective_value))
elif status == OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))




