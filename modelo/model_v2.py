from collections import defaultdict
from parameters import *
from instance_generator import InstanceGenerator
import pickle
import random
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
from tabulate import tabulate
from greedy import run_greedy
from copy import copy

#### FUNCIONES ÚTILES ####


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
    #### CARGAR DATOS ####
    file = open('../datos/servicios.pickle', 'rb')
    services = pickle.load(file)
    file.close()

    file = open('../datos/padres.pickle', 'rb')
    parents = pickle.load(file)
    file.close()

    file = open('../datos/casos.pickle', 'rb')
    cases = pickle.load(file)
    file.close()

    file = open('../datos/abogados.pickle', 'rb')
    unfiltered_lawyers = pickle.load(file)
    file.close()

    file = open('../datos/decodificacion.pickle', 'rb')
    specialties_decod = pickle.load(file)
    file.close()

    file = open('../datos/decod_nombres.pickle', 'rb')
    lawyers_decod = pickle.load(file)
    file.close()

    return services, parents, cases, unfiltered_lawyers, specialties_decod, lawyers_decod


def select_existing_case(cases):
    """
    OUTPUT:
        idx: índice del caso seleccionado tal que cases[idx] es el caso en cuestión
    """
    #### SELECCIONAR CASO A SOLUCIONAR ####
    print(
        f"Seleccionar el caso a resolver: (numero entre el 0 al {len(cases) - 1})\n")
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
    return idx


def show_alternatives(instance, lawyers_decod, specialties_decod):
    S0 = range(instance.S_0)
    services = {s: f"{instance.ids[s]} ({instance.h[s]} h/s)" for s in S0}
    columns = ["Lawyer", "Rating", "Time (h/s)"]
    index = pd.MultiIndex.from_product([services.values(), columns])
    df = pd.DataFrame(columns=index)
    for s in S0:
        s_id = instance.ids[s]
        sorted_l = sorted(
            instance.L, key=lambda x: instance.r[x, instance.ids[s]], reverse=True)
        for i, l in enumerate(sorted_l):
            if instance.r[l, s_id] > 0:
                df.loc[i, services[s]] = l, instance.r[l,
                                                       s_id], instance.d[l, 1]
            else:
                break

    #h = list(map('\n'.join, df.columns.tolist()))
    #print(tabulate(df, tablefmt='psql', headers=h))
    print(df.to_string())


def verify_correctness(model, instance, assignment_mip, time_left_mip, assignment_greedy, time_left_greedy, cases_rating_mip, cases_rating_greedy):
    '''
    Función que verifica la correctitud del modelo de programación lineal y greedy
    '''
    case = instance.base_cases[0]
    rating_mip = cases_rating_mip[0]
    rating_greedy = cases_rating_greedy[0]
    lawyers_time = {}
    for s in case:
        print(f'ASIGNADOS SERVICIO {s}')
        div2 = len(assignment_mip[s])
        print('**MIP**')
        if div2 == 0:
            print('No se asignaron abogados!\n')

        for l in assignment_mip[s]:
            print(f'Abogado N° {l}')
            print(f'Tiempo basal: {instance.d[l, 1]} h/s ')
            print(f'Tiempo destinado: {value(model.t[l, s])} h/s\n')

            if ('mip', l) not in lawyers_time:
                lawyers_time['mip', l] = instance.d[l, 1] - \
                    value(model.t[l, s])
            else:
                lawyers_time['mip', l] -= value(model.t[l, s])

        div = len(assignment_greedy[s])
        print('**GREEDY**')
        if div == 0:
            print('No se asignaron abogados!\n')

        for l in assignment_greedy[s]:
            print(f'Abogado N° {l}')
            print(f'Tiempo basal: {instance.d[l, 1]} h/s ')
            print(f'Tiempo destinado: {instance.h[s] / div} h/s\n')

            if ('greedy', l) not in lawyers_time:
                lawyers_time['greedy', l] = instance.d[l, 1] - \
                    instance.h[s] / div
            else:
                lawyers_time['greedy', l] -= instance.h[s] / div

    print(f'RATING MIP: {rating_mip}')
    print(f'RATING GREEDY: {rating_greedy}\n')

    # Verificar si los tiempos restantes coinciden
    for (est, l) in lawyers_time:
        if est == 'mip':
            time = time_left_mip[l, 1]
        else:
            time = time_left_greedy[l, 1]
        if time != lawyers_time[est, l]:
            print(
                f'Tiempo restante de abogado {l} no coincide usando estrategia {est}!')
            print(f'{time} v/s {lawyers_time[est, l]}\n')


class ILModel:
    def __init__(self, instance):

        # Al inicializarse por primera vez se carga la instancia
        # internamente
        self.charge_instance(instance)

    def charge_instance(self, ins):
        '''
        Función que carga una instancia y genera todas las restricciones
        y variables del modelo
        '''

        #### MODELO ####
        self.model = ConcreteModel()

        #### VARIABLES ####
        self.model.x = Var(ins.L, range(ins.S), domain=Binary)
        self.model.y = Var(range(ins.S), domain=NonNegativeIntegers)
        self.model.t = Var(ins.L, range(ins.S), domain=NonNegativeReals)
        self.model.z = Var(ins.L, range(ins.E + 1),
                           range(1, ins.P + 1), domain=NonNegativeReals)
        self.model.n = Var(range(ins.S), domain=Binary)
        self.model.R = Var(range(ins.S), domain=NonNegativeReals)

        #### FUNCIÓN OBJETIVO ####
        self.model.obj = Objective(expr=sum(self.model.R[s] - ins.beta * (self.model.y[s] - 1) - ins.gamma * self.model.n[s] for s in range(ins.S_0)) +
                                   (1/ins.E) * sum((ins.lambd ** ins.sp[s]) * (self.model.R[s] - ins.beta * (self.model.y[s] - 1) - ins.gamma * self.model.n[s])
                                   for s in range(ins.S_0, ins.S)), sense=maximize)

        #### RESTRICCIONES ####

        self.model.r1 = ConstraintList()
        self.model.r2 = ConstraintList()
        self.model.r3 = ConstraintList()
        self.model.r4 = ConstraintList()
        self.model.r5 = ConstraintList()
        self.model.r6 = ConstraintList()
        self.model.r7 = ConstraintList()

        for s in range(ins.S):
            # R1
            self.model.r1.add(
                sum(self.model.t[l, s] for l in ins.L) == ins.h[s] * (1 - self.model.n[s]))

            # R4
            self.model.r4.add(self.model.y[s] == sum(
                self.model.x[l, s] for l in ins.L))

            # R5
            self.model.r5.add(1 - self.model.y[s] <= self.model.n[s])

            for l in ins.L:
                # R2
                self.model.r2.add(
                    ins.tmin * self.model.x[l, s] <= self.model.t[l, s])
                self.model.r2.add(
                    BIG_M * self.model.x[l, s] >= self.model.t[l, s])

                # R3
                if ins.r[l, ins.ids[s]] == 0:
                    self.model.r3.add(self.model.x[l, s] == 0)

                # R5
                self.model.r5.add(self.model.n[s] <= 1 - self.model.x[l, s])

        # R6
        for l in ins.L:
            for p in range(1, ins.P + 1):
                self.model.r6.add(
                    self.model.z[l, 0, p] == ins.d[l, p] - sum(self.model.t[l, s] for s in ins.active[0, p]))
                for e in range(1, ins.E + 1):
                    self.model.r6.add(
                        self.model.z[l, e, p] == self.model.z[l, 0, p] - sum(self.model.t[l, s] for s in ins.active[e, p]))

        # R7 (definición de R)
        for s in range(ins.S):
            self.model.r7.add(self.model.R[s] == sum(
                self.model.t[l, s] * ins.H[s] * ins.r[l, ins.ids[s]] for l in ins.L))

        # Guardamos la instancia para después
        self.instance = ins

    def run_mip(self, solver='gurobi', time_limit=40):
        '''
        INPUT:
            solver: se especifica que solver utilizar. El default es gurobi
            time_limit: límite de tiempo para resolver el modelo
        OUTPUT:
            assignment: lista de listas tal que assignment[i] es una lista de abogados 
                        asignados al servicio i
            time_left: diccionario de key (l, p) y value tiempo en horas
                       disponibles del abogado l en el periodo p
            time_assigned: diccionario de key (l, p) y value tiempo en horas asignado al abogado 
                           durante el periodo p
            cases_rating: lista tal que cases_rating[j] es el rating de asignación del
                          caso j
        '''
        print('Comenzamos a resolver\n')

        #### ACÁ VA LA CONFIGURACIÓN DEL SOLVER ####
        opt = SolverFactory(solver)
        opt.options['timelimit'] = time_limit
        ###########################################

        opt.solve(self.model)

        # Se construyen estructuras a retornar
        ins = self.instance
        base_cases = ins.base_cases
        assignment = []
        time_left = {}
        time_assigned = defaultdict(int)
        cases_rating = []

        for l in ins.L:
            for p in range(1, ins.P + 1):
                time_left[l, p] = value(self.model.z[l, 0, p])
            for s in range(ins.S_0):
                if value(self.model.x[l, s]):
                    for p in range(1, ins.H[s] + 1):
                        time_assigned[l, p] += value(self.model.t[l, s])

        for case in base_cases:
            of = 0
            for s in case:
                a = []
                of += value(self.model.R[s])
                for l in ins.L:
                    if value(self.model.x[l, s]):
                        a.append(l)
                assignment.append(a)
            cases_rating.append(of)

        return assignment, time_left, time_assigned, cases_rating


if __name__ == "__main__":
    services, parents, cases, unfiltered_lawyers, specialties_decod, lawyers_decod = load_data()
    idx = select_existing_case(cases)

    random.seed(7)
    np.random.seed(7)
    instance = InstanceGenerator(cases, services, unfiltered_lawyers,
                                 parents, NSCENARIOS, RATE, LAMBDA, T_MIN, POND, base_cases=idx)

    show_alternatives(instance, lawyers_decod, specialties_decod)

    model = ILModel(instance)

    assignment_mip, time_left_mip, _, cases_rating_mip = model.run_mip()
    assignment_greedy, time_left_greedy, _, cases_rating_greedy = run_greedy(
        instance, copy(instance.d))

    verify_correctness(model.model, instance, assignment_mip, time_left_mip,
                       assignment_greedy, time_left_greedy, cases_rating_mip, cases_rating_greedy)

    # X = model.model.x.get_values()
    # Y = model.model.y.get_values()
    # T = model.model.t.get_values()
    # Z = model.model.z.get_values()
    # N = model.model.n.get_values()

    # cases_dict = {id: name for id,
    #               name in specialties_decod.items() if id in cases[idx]}
    # assignment = {}

    # for (l, e, p), z in Z.items():
    #     if e < 3 and p < 3 and l < 3:
    #         print(f"Z({l}, {e}, {p}) = {z}")

    # S0 = range(instance.S_0)
    # for s in S0:
    #     s_id = instance.ids[s]
    #     print(f"\n# de abogados asignados a {s_id}: {Y[s]}")
    #     print(f"Tiempo del servicio {s_id}: {instance.h[s]}\n")
    #     assigned_lawyers = []

    #     for l in instance.L:
    #         if X[l, s] == 1:
    #             assigned_lawyers.append(lawyers_decod[l])
    #             print(
    #                 f"Tiempo asignado a {lawyers_decod[l]}: {T[l, s]} / {instance.d[l, 1]}")
    #     assignment[cases_dict[instance.ids[s]]] = assigned_lawyers
    #     if Y[s] == 0:
    #         print(f"Alternativas para {s_id}:")
    #         sorted_l = sorted(
    #             instance.L, key=lambda x: instance.r[x, instance.ids[s]], reverse=True)
    #         for i in range(len(sorted_l)):
    #             print(
    #                 f"{lawyers_decod[sorted_l[i]]}, r = {instance.r[sorted_l[i], instance.ids[s]]}, Z = {Z[sorted_l[i], 0, 1]}")

    # print(assignment)
