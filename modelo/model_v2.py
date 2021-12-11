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

def show_alternatives(instance, save=True, idx="nada"):
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
    print(df)
    return df
    if save:
        df.to_excel(f"../datos/asignaciones/alternativas_caso_{idx}.xlsx ")

class ILModel:
    def __init__(self, instance, name="modelo", mode='SAA'):
        # Al inicializarse por primera vez se carga la instancia
        # internamente
        self.name = name
        self.mode = mode
        if mode=='SAA':
            self.charge_instance(instance)
        else:
            self.charge_greedy(instance)

    def charge_instance(self, ins):
        '''
        Función que carga una instancia y genera todas las restricciones
        y variables del modelo
        '''

        #### MODELO ####
        self.model = ConcreteModel()

        #### VARIABLES ####
        self.model.x = Var(ins.L, range(ins.S), domain=Binary)
        self.model.y = Var(range(ins.S), domain=NonNegativeReals)
        self.model.t = Var(ins.L, range(ins.S), domain=NonNegativeReals)
        self.model.z = Var(ins.L, range(ins.E + 1),
                           range(1, ins.P + 1), domain=NonNegativeReals)
        self.model.n = Var(range(ins.S), domain=Binary)
        self.model.R = Var(range(ins.S), domain=NonNegativeReals)

        #### FUNCIÓN OBJETIVO ####
        self.model.obj = Objective(expr=sum(self.model.R[s] - ins.beta * (self.model.y[s] - 1 + self.model.n[s]) - ins.gamma * self.model.n[s] for s in range(ins.S_0)) +
                                   (1/ins.E) * sum((ins.lambd ** ins.sp[s]) * (self.model.R[s] - ins.beta * (self.model.y[s] - 1 + self.model.n[s]))
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

    def charge_greedy(self, ins):
        """
        Crea un modelo greedy
        """
        #### MODELO ####
        self.model = ConcreteModel()

        #### VARIABLES ####
        self.model.x = Var(ins.L, range(ins.S_0), domain=Binary)
        self.model.y = Var(range(ins.S_0), domain=NonNegativeReals)
        self.model.t = Var(ins.L, range(ins.S_0), domain=NonNegativeReals)
        self.model.n = Var(range(ins.S_0), domain=Binary)
        self.model.R = Var(range(ins.S_0), domain=NonNegativeReals)

        #### FUNCIÓN OBJETIVO ####
        self.model.obj = Objective(expr=sum(self.model.R[s] -  ins.beta * (self.model.y[s] - 1 + self.model.n[s]) - ins.gamma * self.model.n[s] for s in range(ins.S_0)),
                                   sense=maximize)

        #### RESTRICCIONES ####

        self.model.r1 = ConstraintList()
        self.model.r2 = ConstraintList()
        self.model.r3 = ConstraintList()
        self.model.r4 = ConstraintList()
        self.model.r5 = ConstraintList()
        self.model.r6 = ConstraintList()
        self.model.r7 = ConstraintList()

        for s in range(ins.S_0):
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
        #for l in ins.L:
        #    self.model.r6.add(ins.d[l, 1] - sum(self.model.t[l, s] for s in ins.active[0, 1]) >= 0)
            
        # R7 (definición de R)
        for s in range(ins.S_0):
            self.model.r7.add(self.model.R[s] == sum(
                self.model.t[l, s] * ins.H[s] * ins.r[l, ins.ids[s]] for l in ins.L))

        # Guardamos la instancia para después
        self.instance = ins

    def run_mip(self, solver='gurobi', time_limit=60):
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

        if self.mode == "SAA":
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
        else:
            return None, None, None, None

    def show_assignment(self):
        S0 = range(self.instance.S_0)
        services = {s: f"{self.instance.ids[s]} ({self.instance.h[s]} h/s)" 
                    for s in S0}
        columns = ["L", "R", "T (h/s)"]
        index = pd.MultiIndex.from_product([services.values(), columns])
        df = pd.DataFrame(columns=index)
        for s in S0:
            count = 0 # contador de abogados asignados a s
            s_id = self.instance.ids[s]
            for l in self.instance.L:
                if value(self.model.x[l, s]) > 0:
                    count += 1
                    df.loc[count, services[s]] = (l, self.instance.r[l, s_id],
                                            value(self.model.t[l, s]))

        #h = list(map('\n'.join, df.columns.tolist()))
        #print(tabulate(df, tablefmt='psql', headers=h))
        print(" ", "-"*60, f"ASIGNACION {self.mode}", sep="\n")
        print(df)
        print("-"*60)
        return df

    def show_FO(self):
        print(" ", "-"*60, f"DATOS DE LA SOLUCION {self.mode}", sep="\n")
        not_assigned = sum([value(self.model.n[s]) for s in range(self.instance.S_0)])
        total_assigned = 0
        for s in range(self.instance.S_0):
            for l in self.instance.L:
                total_assigned += value(self.model.x[l, s])
        print(f"\n* Valor FO: {value(self.model.obj)}")
        print(f"* Servicios base sin asignar: {not_assigned}")
        print(f"* promedio de abogados asignados por servicio: {value(total_assigned) / self.instance.S_0}")
        base_rating = sum(value(self.model.R[s]) for s in range(self.instance.S_0))
        print(f"* Rating total de la asignacion (sin penalizaciones): {base_rating}")
        penalty = (sum(self.instance.gamma * value(value(self.model.n[s])) 
                      for s in range(self.instance.S_0)) 
                      + self.instance.beta * value(total_assigned))
        print(f"* Rating total de la asignacion (con penalizaciones): {base_rating - penalty}")
        print("-"*60) 
        print(self.instance.beta, self.instance.gamma)
        data = pd.Series({"Rating": base_rating, 
                          "Rating penalizado": base_rating - penalty, 
                          "Servicios sin asignar": not_assigned, 
                          "Promedio l/s": value(total_assigned) / self.instance.S_0})
        return data

    def get_scenarios(self):
        """
        Crea un diccionario con key escenario (int) y value una lista de servicios
        que corresponden a ese escenario (ints).
        """
        active = self.instance.active
        scenarios = {}
        for e in range(1, self.instance.E + 1):
            services = set()
            for p in range(1, self.instance.P):
                services.update(active[e, p])
            scenarios[e] = sorted(list(services))
        return scenarios


    def show_future(self):
        scenarios = self.get_scenarios()
        print(" ", "-"*60, "DATOS ESCENARIOS FUTUROS", sep="\n")


    def show_solution(self):
        assignment = self.show_assignment()
        data = self.show_FO()
        if self.mode=="SAA":
            data2 = self.show_future()
        return assignment, data


        
if __name__ == "__main__":
    services, parents, cases, unfiltered_lawyers, specialties_decod, lawyers_decod = load_data()

    idx = select_existing_case(cases)
    random.seed(7)
    np.random.seed(7)
    instance = InstanceGenerator(cases, services, unfiltered_lawyers,
                                parents, NSCENARIOS, RATE, LAMBDA, T_MIN, POND, base_cases=idx)

    S0 = instance.S_0
    L = instance.L

    alternatives = show_alternatives(instance, idx=idx)

    print("\nCreando modelos...")
    SAA_model = ILModel(instance, name=str(idx))
    greedy_model = ILModel(instance, name=str(idx), mode="GREEDY")
    print("Modelos creado.")
    print("\nEjecutando SAA...")
    SAA_model.run_mip()
    print("Ejecucion terminada.")
    print("\nEjecutando greedy...")
    greedy_model.run_mip()
    print("Ejecucion terminada.")
    greedy_assignment, greedy_data = greedy_model.show_solution()
    SAA_assignment, SAA_data = SAA_model.show_solution()
    with pd.ExcelWriter(f'../asignaciones/caso_{idx}.xlsx') as writer:  
        SAA_assignment.to_excel(writer, sheet_name='Asignacion SAA')
        SAA_data.to_excel(writer, sheet_name="Datos asignacion SAA")
        greedy_assignment.to_excel(writer, sheet_name='Asignacion greedy')
        greedy_data.to_excel(writer, sheet_name="Datos asignacion greedy")
        alternatives.to_excel(writer, sheet_name='Alternativas')

    """
    for idx in range(len(cases)):
        random.seed(7)
        np.random.seed(7)
        instance = InstanceGenerator(cases, services, unfiltered_lawyers,
                                    parents, NSCENARIOS, RATE, LAMBDA, T_MIN, POND, base_cases=idx)

        S0 = instance.S_0
        L = instance.L

        alternatives = show_alternatives(instance, idx=idx)

        print("\nCreando modelo...")
        model = ILModel(instance, name=str(idx))
        print("Modelo creado.")
        print("\nEjecutando modelo...")
        assignment, time_left, time_assigned, service_rating = model.run_mip()
        print("Ejecucion terminada.")
        assignment, data = model.show_solution()
        with pd.ExcelWriter(f'../asignaciones/caso_{idx}.xlsx') as writer:  
            assignment.to_excel(writer, sheet_name='Asignacion')
            data.to_excel(writer, sheet_name="Datos asignacion")
            alternatives.to_excel(writer, sheet_name='Alternativas')
"""    

