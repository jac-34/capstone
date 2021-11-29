# from pyomo.core.base.set import AbstractFiniteSimpleRangeSet
from parameters import *
from instance_generator import InstanceGenerator
import pickle
import random
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
from copy import copy
import pandas as pd
from prettytable import PrettyTable
from os import abort
from tabulate import tabulate

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
    Le pide al usuario un caso de la lista cases. 
    Retorna un int con el índice del caso
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


class ILModel:
    def __init__(self, instance, specialties_decod, lawyers_decod):
        # Cargamos la instancia
        self.charge_instance(instance)

        # La guardamos
        self.instance = instance

        # Guardamos los decodificadores
        self.sd = specialties_decod
        self.ld = lawyers_decod

        # Horas basales para los resultados usando greedy
        self.z_greedy = copy(self.instance.d)

    def charge_instance(self, ins):

        #### MODELO ####
        self.model = ConcreteModel()

        #### VARIABLES ####
        self.model.x = Var(ins.L, range(ins.S), domain=Binary)
        self.model.y = Var(range(ins.S), domain=NonNegativeIntegers)
        self.model.t = Var(ins.L, range(ins.S), domain=NonNegativeReals)
        self.model.z = Var(ins.L, range(1, ins.E + 1),
                           range(1, ins.P + 1), domain=NonNegativeReals)
        self.model.n = Var(range(ins.S), domain=Binary)

        #### FUNCIÓN OBJETIVO ####
        self.model.obj = Objective(expr=sum(sum(self.model.t[l, s] * ins.H[s] * ins.r[l, ins.ids[s]] for l in ins.L) -
                                   ins.beta * (self.model.y[s] - 1) - ins.gamma * self.model.n[s] for s in range(ins.S_0)) +
                                   (1/ins.E) * sum((ins.lambd ** ins.sp[s]) * (sum(self.model.t[l, s] * ins.H[s] * ins.r[l, ins.ids[s]] for l in ins.L) -
                                                                               ins.beta * (self.model.y[s] - 1) - ins.gamma * self.model.n[s]) for s in range(ins.S_0, ins.S)), sense=maximize)

        #### RESTRICCIONES ####

        self.model.r1 = ConstraintList()
        self.model.r2 = ConstraintList()
        self.model.r3 = ConstraintList()
        self.model.r4 = ConstraintList()
        self.model.r5 = ConstraintList()
        self.model.r6 = ConstraintList()

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

        for l in ins.L:
            for e in range(1, ins.E + 1):
                for p in range(1, ins.P + 1):
                    # R6
                    self.model.r6.add(
                        self.model.z[l, e, p] == ins.d[l, p] - sum(self.model.t[l, s] for s in (ins.active[e, p] + ins.active[0, p])))

    def run_mip(self, solver='gurobi'):
        print('Comenzamos a resolver')
        opt = SolverFactory(solver)
        opt.options['timelimit'] = 40
        opt.solve(self.model)

    def has_time(self, l, s, div):
        '''
        Función que verifica si el abogado l tiene tiempo para
        realizar el servicio s
        '''

        for p in range(1, self.instance.H[s] + 1):
            if self.z_greedy[l, p] < self.instance.h[s] / div or self.instance.h[s] < self.instance.tmin:
                return False
        return True

    def run_greedy(self):

        # greedy_assignment[i] es una lista de abogados asignados al servicio i
        self.greedy_assignment = []

        # greedy_of[i] es el rating de asignación del caso i
        self.greedy_of = []

        base_cases = self.instance.base_cases
        for case in base_cases:
            of = 0
            for s in case:
                s_id = self.instance.ids[s]
                filt = list(
                    filter(lambda x: self.instance.r[x, s_id] > 0, self.instance.L))
                ordered_lawyers = sorted(
                    filt, key=lambda x: self.instance.r[x, s_id], reverse=True)
                div = 1
                sol = False
                candidates = []
                while self.instance.h[s] / div >= self.instance.tmin:
                    if div > len(ordered_lawyers):
                        break
                    for idx in range(len(ordered_lawyers)):
                        if self.has_time(ordered_lawyers[idx], s, div):
                            candidates.append(ordered_lawyers[idx])
                        if len(candidates) == div:
                            sol = True
                            break
                    if sol:
                        break
                    div += 1
                    candidates = []

                for l in candidates:
                    for p in range(1, self.instance.H[s] + 1):
                        self.z_greedy[l, p] -= self.instance.h[s] / div
                        of += (self.instance.h[s] * self.instance.H[s]
                               * self.instance.r[l, s_id]) / div
                self.greedy_assignment.append(candidates)
            self.greedy_of.append(of)

    def greedy_results(self, max_cases=3):
        '''
        Se imprime una tabla con los resultados de greedy

        max_cases indica hasta cuántos casos se quiere imprimir
        '''
        limit = min(max_cases, len(self.instance.base_cases))
        for idc in range(limit):
            case = self.instance.base_cases[idc]
            col_len = max([len(self.greedy_assignment[s]) for s in case])
            table = PrettyTable()
            table.title = f'CASO {idc + 1}'
            for s in case:
                col = [self.ld[l] for l in self.greedy_assignment[s]]
                fill = [''] * (col_len - len(col))
                table.add_column(
                    f'{self.instance.ids[s]}: {self.instance.h[s]} h/s', col + fill)
            print(table)
            print(f'Rating: {self.greedy_of[idc]}')

    def reboot_model(self):
        pass

def show_alternatives(instance, lawyers_decod, specialties_decod):
    S0 = range(instance.S_0)
    services = {s: f"{instance.ids[s]} ({instance.h[s]} h/s)" for s in S0}
    columns = ["Lawyer", "Rating", "Time (h/s)"]
    index = pd.MultiIndex.from_product([services.values(), columns])
    df = pd.DataFrame(columns=index)
    for s in S0:
        s_id = instance.ids[s]
        sorted_l = sorted(instance.L, key= lambda x: instance.r[x, instance.ids[s]], reverse=True)
        for i, l in enumerate(sorted_l):
            if instance.r[l, s_id] > 0:
                df.loc[i, services[s]] = l, instance.r[l, s_id], instance.d[l, 1]
            else:
                break

    #h = list(map('\n'.join, df.columns.tolist()))
    #print(tabulate(df, tablefmt='psql', headers=h))
    print(df.to_string())
        
            


if __name__ == "__main__":
    services, parents, cases, unfiltered_lawyers, specialties_decod, lawyers_decod = load_data()
    idx = select_existing_case(cases)

    random.seed(40)
    np.random.seed(40)
    instance = InstanceGenerator(cases, services, unfiltered_lawyers,
                                 parents, NSCENARIOS, RATE, LAMBDA, T_MIN, base_cases=idx)

    show_alternatives(instance, lawyers_decod, specialties_decod)

    model = ILModel(instance, specialties_decod, lawyers_decod)
    model.run_greedy()
    model.greedy_results()
    abort()
    model.run_mip()

    X = model.model.x.get_values()
    Y = model.model.y.get_values()
    T = model.model.t.get_values()
    Z = model.model.z.get_values()
    N = model.model.n.get_values()

    cases_dict = {id: name for id,
                  name in specialties_decod.items() if id in cases[idx]}
    assignment = {}

    for (l, e, p), z in Z.items():
        if e < 3 and p < 3 and l < 3:
            print(f"Z({l}, {e}, {p}) = {z}")

    S0 = range(instance.S_0)
    for s in S0:
        s_id = instance.ids[s]
        print(f"\n# de abogados asignados a {s_id}: {Y[s]}")
        print(f"Tiempo del servicio {s_id}: {instance.h[s]}\n")
        assigned_lawyers = []

        for l in instance.L:
            if X[l, s] == 1:
                assigned_lawyers.append(lawyers_decod[l])
                print(f"Tiempo asignado a {lawyers_decod[l]}: {T[l, s]} / {instance.d[l, 1]}")
        assignment[cases_dict[instance.ids[s]]] = assigned_lawyers
        if Y[s] == 0:
            print(f"Alternativas para {s_id}:")
            sorted_l = sorted(instance.L, key= lambda x: instance.r[x, instance.ids[s]], reverse=True)
            for i in range(len(sorted_l)):
                print(f"{lawyers_decod[sorted_l[i]]}, r = {instance.r[sorted_l[i], instance.ids[s]]}, Z = {Z[sorted_l[i], 0, 1]}")

    print(assignment)
