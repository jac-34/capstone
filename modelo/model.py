from generador_instancia import GeneradorInstancia
from rating_function import process_ratings, global_register
from parameters import *
import pickle
from mip import Model, xsum, MAXIMIZE, BINARY, CONTINUOUS, CBC, maximize, OptimizationStatus
import random
from greedy import greedy
import numpy as np
from os import abort

def load_data():
    """
    Carga los datos \\
    Output:
        services: pd.DataFrame con info de los servicios.
        parents: lista, donde parents[s_id] corresponde al "padre" del servicio s_id.
        cases: lista de listas, donde cada lista tiene ints correspondiente a ids de los
        servicios.
        unfiltered_lawyers: pd.DataFrame, con la info de los abogados.
        specialties_decod: dict con key (s_id) y value str (el nombre del servicio s_id)
        lawyers_decod: dict con key (l_id) y value str (el nombre del abogado l_id)
    """
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
    unfiltered_lawyers = pickle.load(file)
    file.close()

    file = open('decodificacion.pickle', 'rb')
    specialties_decod = pickle.load(file)
    file.close()

    file = open('decod_nombres.pickle', 'rb')
    lawyers_decod = pickle.load(file)
    file.close()
    return services, parents, cases, unfiltered_lawyers, specialties_decod, lawyers_decod

def select_existing_case(cases):
    """
    Le pide al usuario un caso de la lista cases. 
    Retorna una lista case = [s_id1, ..., s_idn] con los ids del caso.
    """
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
    return case

def pre_process_data(case, cases, services, unfiltered_lawyers, parents, 
                     specialties_decod):
    """
    Pre procesa los datos
    """
    ### GENERAR CASOS FUTUROS ###
    instance = GeneradorInstancia(cases, services, unfiltered_lawyers, parents, N=200)
    service_classes, active, lawyers = instance.inicializar_generador(base_case=case)

    ### REVISAR LA GENERACION DE LOS CONJUNTOS P Y E!!! ###
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
    ## evaluar poner algo mas rapido ##
    S_arrival = {(e, p): [s.ngen for s in service_classes if s.escenario == e and s.periodo == p]
                for e in E for p in P}

    register = global_register(services, parents, lawyers)

    ### PARAMETROS ###
    r = process_ratings(lawyers, s_ids.values(), parents, register, depth=3) #rating
    d = lawyers["hb"].to_dict()
    h = {s.ngen: s.hsemanas for s in service_classes}
    H = {s.ngen: s.semanas for s in service_classes}

    ### AQUI SE PODRIA IMPRIMIR LA TABLA CON LOS PRIMEROS ABOGADOS
    for l, s in r.keys():
        if s in case_idx:
            print(f"r[{l}, {s}]", r[l, s])
    
    return P, E, L, S, S0, S_arrival, active, r, s_ids, d, h, H

def create_model(S, S0, P, E, L, S_arrival, active, r, s_ids, d, h, H, 
                 BETA, GAMMA, LAMBDA, M=10**8):
    """
    Crea el modelo de asignacion de abogados.
    -----------------------------------------------
    INPUT:
        CONJUNTOS:
        S: lista con los ngen de todos los servicios.
        S0: lista con los ngen de los servicios del caso base.
        P: lista de periodos.
        E: lista de escenarios.
        L: lista de abogados.\\
        PARAMETROS:\\
        S_arrival: dict con key (escenario, periodo) y value [ngen1, ...]
        una lista de los ngen de los servicios que llegan en ese escenario en ese periodo.
        active: dict con key (escenario, periodo) y value [ngen1, ...] una lista de los
        ngen de los servicios que estan activos en ese escenario en ese periodo
        r: dict con key (l_id, s_id) y value float que da el rating del abogado l_id
        para el servicio s_id.
        s_ids: dict con key ngen (int) y value s_id (int). El value corresponde al id
        del servicio correspondiente a ngen.
        d: dict con key l_id (int) y value int, que corresponde a cuantas hrs/semana 
        disponibles tiene el abogado l_id.
        h: dict con key ngen (int) y value int, que corresponde a cuantas hrs/semana
        requiere el servicio ngen.
        H: dict con key ngen (int) y value int, que corresponde a cuantas semanas dura
        el servicio ngen.
        BETA: parametro
        GAMMA: parametro
        LAMBDA: parametro
        M: Valor arbitrariamente grande

    OUTPUT:
        m: objeto mip.Model
        x: dict con key (l_id, ngen), value mip.Var.
        y: dict con key (ngen), value mip.Var.
        t: dict con key (l_id, s_id), value mip.Var.
        z: dict con key (l_id, e (escenario), p (periodo)), value mip.Var.
        n: dict con key (s_id), value mip.Var
    """
    ### MODELO ###
    print("Creando modelo...")
    m = Model(name="Asignacion de abogados", sense=MAXIMIZE, solver_name=CBC)

    ## VARIABLES ##
    x = {(l, s): m.add_var(name=f"x_{l}_{s}", var_type=BINARY) for s in S for l in L}
    y = [m.add_var(name=f"y_{s}", var_type=CONTINUOUS) for s in S]
    t = {(l, s): m.add_var(name=f"t_{l}_{s}", var_type=CONTINUOUS) for s in S for l in L}
    z = {(l, e, p): m.add_var(name=f"z_{l}_{e}_{p}", lb=0.0) for e, p in active.keys() for l in L}
    n = [m.add_var(name=f"n_{s}", lb=0, var_type=BINARY) for s in S]

    ## FUNCION OBJETIVO ##
    m.objective = maximize(
        xsum(H[s] * xsum(t[l, s] * r[l, s_ids[s]] for l in L) - BETA * (y[s] - 1) * H[s] * h[s]  - GAMMA * n[s] 
        for s in S0) + 
        1 / len(E) * xsum((LAMBDA ** p) * xsum(H[s] * xsum(t[l, s] * r[l, s_ids[s]] 
        for l in L) - BETA * (y[s] - 1) * H[s] * h[s] - GAMMA * n[s] 
        for s in S_arrival[e, p]) for e, p in S_arrival.keys())
    )
    ## RESTRICCIONES ##
    for s in S:
        # R1 #
        m += xsum(t[l, s] for l in L) == h[s] * (1 - n[s]), f"r1_{s}"
        # R4 #
        m += y[s] == xsum(x[l, s] for l in L), f"r4_{s}"
        # R5 #
        m += n[s] >= 1 - y[s], f"r5_{s}"
        for l in L:
            # R2 #
            m += T_MIN * x[l, s] <= t[l, s], f"r2.1_{l}_{s}"
            m += M * x[l, s] >= t[l, s], f"r2.2_{l}_{s}"
            # R3 #
            if r[l, s_ids[s]] == 0:
                m += x[l, s] == 0, f"r3_{l}_{s}"
    for l in L:
        for e, p in active.keys():
            # R6 #
            m += z[l, e, p] == d[l] - xsum(t[l, s] for s in active[e, p]), f"R6_{l}_{e}_{p}"

    print("Modelo creado.\n")
    m.write("asignacion_abogados.lp")
    return m, x, y, t, z, n

def solve_model(m, max_gap=3, max_seconds=150):
    """
    Resuelve el modelo m
    INPUT:
        m: mip.Model
        max_gap: float
        max_seconds: float
    """
    m.max_gap = max_gap
    status = m.optimize(max_seconds=max_seconds)
    if status == OptimizationStatus.OPTIMAL:
        print('optimal solution cost {} found'.format(m.objective_value))
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))

if __name__ == "__main__":
    random.seed(10)
    
    services, parents, cases, unfiltered_lawyers, specialties_decod, lawyers_decod = load_data()
    case = select_existing_case(cases)
    P, E, L, S, S0, S_arrival, active, r, s_ids, d, h, H = pre_process_data(case, cases, services, unfiltered_lawyers,
                                                             parents, specialties_decod)

    m, x, y, t, z, n = create_model(S, S0, P, E, L, S_arrival, active, r, s_ids, d, h, H, 
                 BETA, GAMMA, LAMBDA, M=10**8)

    solve_model(m)
    
    for s in S:
        if s in S0:
            print(f"\n# de abogados asignados a {s_ids[s]}: {y[s].x}")
            print(f"Tiempo del servicio {s_ids[s]}: {h[s]}\n")
            total_time = 0
            for l in L:
                var = x[l, s]
                if var.x:
                    if var.x > 0:
                        print(f"x_({l}, {s_ids[s]}) = {var.x}")
                        print(f"Tiempo del abogado {l}: {d[l]}")
                        print(f"Tiempo asignado: {t[l, s].x}")
                        total_time += t[l, s].x
            if total_time <= h[s]:
                print(f"variable de penalizacion: n[{s}] = {n[s].x}")
                print(f"tiempo no asignado: {h[s] - total_time}")
            if y[s].x == 0:
                print(f"Alternativas para {s_ids[s]}:")
                sorted_l = sorted(L, key= lambda x: r[x, s_ids[s]], reverse=True)
                for i in range(len(sorted_l)):
                    print(f"l: {sorted_l[i]}, r = {r[sorted_l[i], s_ids[s]]}, z = {z[sorted_l[i], 0, 1].x}")
    fo_solver = sum(H[s] * sum(t[l, s].x * r[l, s_ids[s]] for l in L) for s in S0)
    print(f"FO del solver: {fo_solver}")


    greedy_sol, time, of = greedy(L, S0, s_ids, r, d, h, H)
    print("-------SOLUCION GREEDY-------")
    print(f"Valor funcion objetivo: {of}")
    for ngen, l_id in greedy_sol.items():
        print(f"Servicio {s_ids[ngen]} asignado a {l_id}. {h[ngen]} hrs por {H[ngen]} semanas.")

