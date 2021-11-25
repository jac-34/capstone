from parameters import *
from instance_generator import InstanceGenerator
import pickle
import random
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory


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
    def __init__(self, instance):
        # Cargamos la instancia
        self.charge_instance(instance)

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
                        self.model.z[l, e, p] == ins.d[l] - sum(self.model.t[l, s] for s in ins.active[e, p]))

    def run(self, solver='gurobi'):
        print('comenzamos a resolver')
        opt = SolverFactory(solver)
        opt.solve(self.model)
        print(value(self.model.obj))


if __name__ == "__main__":
    services, parents, cases, unfiltered_lawyers, _, _ = load_data()
    case = select_existing_case(cases)

    random.seed(40)
    np.random.seed(40)
    instance = InstanceGenerator(cases, services, unfiltered_lawyers,
                                 parents, NSCENARIOS, RATE, LAMBDA, T_MIN, base_case=case)
    model = ILModel(instance)
    model.run()
