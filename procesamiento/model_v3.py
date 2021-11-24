from mip.constants import INTEGER
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
    #### CARGAR DATOS ####
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
    case = cases[idx]
    return case


class ILModel:
    def __init__(self, case, services, parents, cases, unfiltered_lawyers, beta=BETA, gamma=GAMMA, lambd=LAMBDA, nscenarios=NSCENARIOS, tmin=T_MIN):
        #### PARÁMETROS INICIALES ####
        self.beta = beta
        self.gamma = gamma
        self.lambd = lambd
        
        # Número de escenarios
        self.E = nscenarios
        self.tmin = tmin

        # Creamos la instancia junto al modelo
        self.create_model(case, cases, services, unfiltered_lawyers, parents)

    def create_instance(self, case, cases_list, services_list, unfiltered_lawyers, parents):
        instance = GeneradorInstancia(
            cases_list, services_list, unfiltered_lawyers, parents, N=self.E)

        instance.inicializar_generador(base_case=case)

        ##### PARÁMETROS ####

        # Número de periodos
        self.P = instance.horizonte

        # Conjunto de índices de abogados
        self.L = [l for l in instance.tabla_abogados["id"]]
        

        # Número de servicios generados
        self.S = instance.generados

        # Número de servicios del caso base
        self.S_0 = instance.generados_base

        # Id de cada servicio (asociado al tipo de servicio)
        self.ids = [s.id for s in instance.servicios]

        # Periodo al que pertenece cada servicio
        self.sp = [s.periodo for s in instance.servicios]

        # horas/semana
        self.h = [s.hsemanas for s in instance.servicios]

        # semanas
        self.H = [s.semanas for s in instance.servicios]

        # activos
        self.activos = instance.activos

        # disponibilidad basal abogados
        self.d = {l: list(instance.tabla_abogados[instance.tabla_abogados["id"] == l]["hb"])[
            0] for l in self.L}

        # rating
        register = global_register(
            services_list, parents, instance.tabla_abogados)
        self.r = process_ratings(instance.tabla_abogados, list(
            set(self.ids)), parents, register, depth=100)
        print(self.r)

    def create_model(self, case, cases_list, services_list, unfiltered_lawyers, parents):
        self.create_instance(case, cases_list, services_list,
                             unfiltered_lawyers, parents)

        #### MODELO ####
        self.model = Model(name="Asignacion de abogados",
                           sense=MAXIMIZE, solver_name=CBC)

        #### VARIABLES ####
        self.x = {(l, s): self.model.add_var(var_type=BINARY)
                  for l in self.L for s in range(self.S)}
        self.y = [self.model.add_var(var_type=INTEGER) for s in range(self.S)]
        self.t = {(l, s): self.model.add_var(var_type=CONTINUOUS)
                  for l in self.L for s in range(self.S)}
        self.z = {(l, e, p): self.model.add_var(var_type=CONTINUOUS)
                  for l in self.L for e in range(1, self.E + 1) for p in range(1, self.P + 1)}
        self.n = [self.model.add_var(var_type=BINARY) for s in range(self.S)]

        #### FUNCIÓN OBJETIVO ####
        self.model.objective = maximize(xsum(xsum(self.t[l, s] * self.H[s] * self.r[l, self.ids[s]] for l in self.L) - self.beta * (self.y[s] - 1) - self.gamma * self.n[s] for s in range(self.S_0)) + (1/self.E) * xsum(
            (self.lambd ** self.sp[s]) * (xsum(self.t[l, s] * self.H[s] * self.r[l, self.ids[s]] for l in self.L) - self.beta * (self.y[s] - 1) - self.gamma * self.n[s]) for s in range(self.S_0, self.S)))

        #### RESTRICCIONES ####

        for s in range(self.S):
            # R1
            self.model.add_constr(
                xsum(self.t[l, s] for l in self.L) == self.h[s] * (1 - self.n[s]))

            # R4
            self.model.add_constr(self.y[s] == xsum(
                self.x[l, s] for l in self.L))

            # R5
            self.model.add_constr(self.n[s] >= 1 - self.y[s])

            for l in self.L:
                # R2
                self.model.add_constr(self.tmin * self.x[l, s] <= self.t[l, s])
                self.model.add_constr(BIG_M * self.x[l, s])

                # R3
                if self.r[l, self.ids[s]] == 0:
                    self.model.add_constr(self.x[l, s] == 0)

                # R7
                self.model.add_constr(self.n[s] <= 1 - self.x[l, s])

        for l in self.L:
            for e in range(1, self.E + 1):
                for p in range(1, self.P + 1):
                    self.model.add_constr(
                        self.z[l, e, p] == self.d[l] - xsum(self.t[l, s] for s in self.activos[e, p]))
    
    def run(self):
        status = self.model.optimize()
        if status == OptimizationStatus.OPTIMAL:
            print('Optimal Solution Found')


if __name__ == "__main__":
    services, parents, cases, unfiltered_lawyers, specialties_decod, lawyers_decod = load_data()
    case = select_existing_case(cases)

    model = ILModel(case, services, parents, cases, unfiltered_lawyers, specialties_decod, lawyers_decod)

    model.run()


