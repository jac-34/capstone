from generador_instancia import GeneradorInstancia
from rating_function import process_ratings, global_register
from parameters import *
import pickle
from mip import Model, xsum, MAXIMIZE, BINARY, CONTINUOUS, CBC, maximize, OptimizationStatus
import random
from greedy import greedy

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

print(f"Seleccionar el caso a resolver: (numero entre el 0 al {len(cases) - 1})\n")
while True:
    idx = input("Ingrese un numero: ")
    if idx.isdigit():
        if int(idx) < len(cases):
            break
        else:
            print(f"El numero debe estar entre 0 y {len(cases) - 1}")
    else:
        print("Se debe ingresar un numero")
idx = int(idx)
case = cases[idx]


### GENERAR CASOS FUTUROS ###
instance = GeneradorInstancia(cases, services, lawyers, parents, N=200)
service_classes, active, table = instance.inicializar_generador(base_case=case)

P = [i for i in range(1, instance.horizonte + 1)] # conjunto de periodos
E = [i for i in range(1, instance.n_escenarios + 1)] # conjunto de escenarios
L = [id for id in lawyers["id"]] # conjunto de abogados

# active[scenario, period] = [active cases]
s_ids = {s.ngen: s.id for s in service_classes}
case_idx = [s.id for s in service_classes if s.escenario == 0]
print("Caso a trabajar:")
for id in case_idx:
    print(f"   {id}: {specialties_decod[id]}")
print()

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

x = [[m.add_var(name=f"x_{l}_{s}", var_type=BINARY) for s in S] for l in L]
y = [m.add_var(name=f"y_{s}", var_type=CONTINUOUS) for s in S]
t = [[m.add_var(name=f"t_{l}_{s}", var_type=CONTINUOUS) for s in S] for l in L]
z = {(l, e, p): m.add_var(name=f"z_{l}_{e}_{p}", lb=0.0) for e, p in active.keys() for l in L}
n = [m.add_var(name=f"n_{s}", lb=0) for s in S]

## FUNCION OBJETIVO ##
m.objective = maximize(
    xsum(H[s] * xsum(t[l][s] * r[l, s_ids[s]] for l in L) - BETA * (y[s] - 1)  - GAMMA * n[s] * H[s] * h[s] 
    for s in S0) + 
    1 / len(E) * xsum((LAMBDA ** p) * xsum(H[s] * xsum(t[l][s] * r[l, s_ids[s]] 
    for l in L) - BETA * (y[s] - 1)  - GAMMA * n[s] * H[s] * h[s]
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
        m += T_MIN * x[l][s] <= t[l][s], f"r2.1_{l}_{s}"
        m += M * x[l][s] >= t[l][s], f"r2.2_{l}_{s}"

        # R3 #
        if r[l, s_ids[s]] == 0:
            m += x[l][s] == 0, f"r3_{l}_{s}"

# R6 #
#for l in L:
#    for e in E:
#        for p in P:
#            m += z[l, e, p] == d[l] - xsum(t[l][s] for s in active[e, p]), f"R6_{l}_{e}_{p}"

for l in L:
    for e, p in active.keys():
        m += z[l, e, p] == d[l] - xsum(t[l][s] for s in active[e, p]), f"R6_{l}_{e}_{p}"

print("Modelo creado.\n")
m.write("asignacion_abogados.lp")

m.max_gap = 3
status = m.optimize(max_seconds=150)
if status == OptimizationStatus.OPTIMAL:
    print('optimal solution cost {} found'.format(m.objective_value))
            
elif status == OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))


for s in range(len(x[l])):
    if s in S0:
        for l in range(len(x)):
            var = x[l][s]
            if var.x > 0:
                print(f"x_({l}, {s_ids[s]}) = {var.x}")
                print(f"Tiempo del abogado {l}: {d[l]}")
                print(f"Tiempo del servicio {s_ids[s]}: {h[s]}")
                print(f"Tiempo asignado: {t[l][s].x}") 
                print(f"spare time: {d[l] - t[l][s].x}")
        print(f"n# de abogados asignados a {[s_ids[s]]}: {y[s].x}\n")
fo_solver = sum(H[s] * sum(t[l][s].x * r[l, s_ids[s]] for l in L) for s in S0)
print(f"FO del solver: {fo_solver}")


greedy_sol, time, of = greedy(lawyers["id"], S0, s_ids, r, d, h, H)
print("-------SOLUCION GREEDY-------")
print(f"Valor funcion objetivo: {of}")
for ngen, l_id in greedy_sol.items():
    print(f"Servicio {s_ids[ngen]} asignado a {l_id}. {h[ngen]} hrs por {H[ngen]} semanas.")

