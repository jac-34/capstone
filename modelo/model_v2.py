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

    def run_mip(self, solver='gurobi', time_limit=50):
        '''
        INPUT:
            solver: se especifica que solver utilizar. El default es gurobi
            time_limit: límite de tiempo para resolver el modelo
        OUTPUT:
            assignment: lista de listas tal que assignment[i] es una lista de abogados 
                        asignados al servicio i
            time_left: diccionario de key (l, p) y value tiempo en horas
                       disponibles del abogado l en el periodo p
            time_assigned: diccionario de key (l, p) y value tiempo en horas asignado al abogado l
                           durante el periodo p
            services_rating: lista tal que services_rating[i] es el rating de asignación del
                             servicio i
        '''
        #### ACÁ VA LA CONFIGURACIÓN DEL SOLVER ####
        opt = SolverFactory(solver)
        opt.options['timelimit'] = time_limit
        ###########################################

        opt.solve(self.model)

        # Se construyen estructuras a retornar
        ins = self.instance
        assignment = []
        time_left = {}
        time_assigned = defaultdict(int)
        services_rating = []

        for l in ins.L:
            for p in range(1, ins.P + 1):
                time_left[l, p] = value(self.model.z[l, 0, p])
            for s in range(ins.S_0):
                if value(self.model.x[l, s]):
                    for p in range(1, ins.H[s] + 1):
                        time_assigned[l, p] += value(self.model.t[l, s])

        for s in range(ins.S_0):
            a = []
            services_rating.append(value(self.model.R[s]))
            for l in ins.L:
                if value(self.model.x[l, s]):
                    a.append(l)
            assignment.append(a)

        return assignment, time_left, time_assigned, services_rating


if __name__ == "__main__":
    services, parents, cases, unfiltered_lawyers, specialties_decod, lawyers_decod = load_data()
    idx = select_existing_case(cases)

    random.seed(7)
    np.random.seed(7)
    instance = InstanceGenerator(cases, services, unfiltered_lawyers,
                                 parents, NSCENARIOS, RATE, LAMBDA, T_MIN, POND, base_cases=idx)

    show_alternatives(instance, lawyers_decod, specialties_decod)
