
from collections import defaultdict

from pyomo.core.util import quicksum
from rating_function import process_ratings, global_register
from parameters import *
from gurobipy import *


class GreedyInstanceGenerator:

    def __init__(self, services, lawyers, parents, tmin, pond, base_cases):

        # DataFrame de servicios
        self.services = services

        # DataFrame de abogados
        self.lawyers = lawyers

        # Árbol de especializaciones
        self.parents = parents

        # Número de servicios del caso (o casos) base
        self.S_0 = 0

        # Horizonte de tiempo
        self.P = 0

        # Horas semana
        self.h = []

        # Semanas
        self.H = []

        # Abogados
        self.L = []

        # Diccionario de servicios activos
        self.active = defaultdict(list)

        # Disponibilidades base de abogados
        self.d = {}

        # Id de cada servicio generado
        self.ids = []

        # Otros parámetros
        self.tmin = tmin
        self.pond = pond

        # Lista de casos base (cada caso es una lista con los números de gen de los servicios)
        self.base_cases = []

        self.create_instance(base_cases)

    def service_area(self, id):
        '''
        OUTPUT:
            current: área a la que pertenece el servicio con identificador
                     id
        '''
        current = id
        while self.parents[current] != -1:
            current = self.parents[current]
        return current

    def create_instance(self, base_cases):
        '''
        Se genera la instancia a utilizar en el modelo
        '''
        # Aprovechamos de calcular beta
        self.beta = 0

        # Se identifican las áreas y se guardan los
        # servicios base
        base_areas = set()
        for case in base_cases:
            c = []
            for id in case:
                hweeks, weeks = case[id]
                area = self.service_area(id)
                base_areas.add(area)
                self.h.append(hweeks)
                self.H.append(weeks)
                self.ids.append(id)
                c.append(self.S_0)
                if weeks > self.P:
                    self.P = weeks
                if hweeks * weeks > self.beta:
                    self.beta = hweeks * weeks
                for per in range(1, weeks + 1):
                    self.active[per].append(self.S_0)
                self.S_0 += 1
            self.base_cases.append(c)

        # Se dividen los abogados en áreas
        lawyers_to_areas = defaultdict(list)
        for row in self.lawyers.itertuples(index=False):
            for area in row.areas:
                lawyers_to_areas[area].append(row.id)

        # Se pobla lista de abogados a considerar
        L = set()
        for area in lawyers_to_areas:
            if area in base_areas:
                for l in lawyers_to_areas[area]:
                    L.add(l)
                    for p in range(1, self.P + 1):
                        self.d[l, p] = list(
                            self.lawyers[self.lawyers["id"] == l]["hb"])[0]
        self.L = list(L)

        # Se identifican abogados
        lawyers = self.lawyers[self.lawyers['id'].isin(self.L)]

        # Aprovechamos de calcular gamma
        self.gamma = self.pond * self.beta

        # Calculamos rating
        register = global_register(self.services, self.parents, lawyers)
        self.r = process_ratings(lawyers, list(
            set(self.ids)), self.parents, register, depth=100)

    def reboot_instance(self, d, base_cases):
        '''
        Función que reinicia la instancia (útil para simulación)
        '''
        self.S_0 = 0
        self.P = 0
        self.h = []
        self.H = []
        self.L = []
        self.active = defaultdict(list)
        self.ids = []
        self.base_cases = []

        # diccionario previamente modificado
        self.d = d

        # Aprovechamos de calcular beta
        self.beta = 0

        # Se identifican áreas base
        base_areas = set()
        for case in base_cases:
            c = []
            for id in case:
                hweeks, weeks = case[id]
                area = self.service_area(id)
                base_areas.add(area)
                self.h.append(hweeks)
                self.H.append(weeks)
                self.ids.append(id)
                c.append(self.S_0)
                if weeks > self.P:
                    self.P = weeks
                if hweeks * weeks > self.beta:
                    self.beta = hweeks * weeks
                for per in range(1, weeks + 1):
                    self.active[per].append(self.S_0)
                self.S_0 += 1
            self.base_cases.append(c)

        # Se dividen los abogados en áreas
        lawyers_to_areas = defaultdict(list)
        for row in self.lawyers.itertuples(index=False):
            for area in row.areas:
                lawyers_to_areas[area].append(row.id)

        # Se pobla lista de abogados a considerar
        L = set()
        for area in lawyers_to_areas:
            if area in base_areas:
                for l in lawyers_to_areas[area]:
                    L.add(l)
                    for p in range(1, self.P + 1):
                        if (l, p) not in self.d:
                            self.d[l, p] = list(
                                self.lawyers[self.lawyers["id"] == l]["hb"])[0]
        self.L = list(l)

        # Se actualiza tabla de abogados
        lawyers = self.lawyers[self.lawyers['id'].isin(self.L)]

        # Aprovechamos de calcular gamma
        self.gamma = self.pond * self.beta

        # Calculamos rating
        register = global_register(self.services, self.parents, lawyers)
        self.r = process_ratings(lawyers, list(
            set(self.ids)), self.parents, register, depth=100)


class GreedyILModel:

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
        self.model = Model('Greedy')

        #### VARIABLES ####
        self.x = self.model.addVars(ins.L, range(ins.S_0), vtype=GRB.BINARY)
        self.y = self.model.addVars(range(ins.S_0), vtype=GRB.INTEGER)
        self.t = self.model.addVars(
            ins.L, range(ins.S_0), vtype=GRB.CONTINUOUS)
        self.z = self.model.addVars(ins.L, range(
            1, ins.P + 1), vtype=GRB.CONTINUOUS)
        self.n = self.model.addVars(range(ins.S_0), vtype=GRB.BINARY)
        self.R = self.model.addVars(range(ins.S_0), vtype=GRB.CONTINUOUS)

        self.model.setParam('OutputFlag', 0)
        self.model.setParam('Presolve', 2)
        self.model.setParam('TimeLimit', 40)
        self.model.update()

        #### FUNCIÓN OBJETIVO ####
        obj = quicksum(self.R[s] - ins.beta * (self.y[s] - 1) -
                       ins.gamma * self.n[s] for s in range(ins.S_0))
        self.model.setObjective(obj, GRB.MAXIMIZE)

        #### RESTRICCIONES ####

        # R1
        self.model.addConstrs(quicksum(
            self.t[l, s] for l in ins.L) == ins.h[s] * (1 - self.n[s]) for s in range(ins.S_0))

        # R2
        self.model.addConstrs(
            ins.tmin * self.x[l, s] <= self.t[l, s] for s in range(ins.S_0) for l in ins.L)
        self.model.addConstrs(
            BIG_M * self.x[l, s] >= self.t[l, s] for s in range(ins.S_0) for l in ins.L)

        # R3
        for s in range(ins.S_0):
            for l in ins.L:
                if ins.r[l, ins.ids[s]] == 0:
                    self.model.addConstr(self.x[l, s] == 0)

        # R4
        self.model.addConstrs(
            self.y[s] == quicksum(self.x[l, s] for l in ins.L) for s in range(ins.S_0))

        # R5
        self.model.addConstrs(1 - self.y[s] <= self.n[s]
                              for s in range(ins.S_0))
        self.model.addConstrs(self.n[s] <= 1 - self.x[l, s]
                              for s in range(ins.S_0) for l in ins.L)

        # R6
        for l in ins.L:
            for p in range(1, ins.P + 1):
                self.model.addConstr(
                    self.z[l, p] == ins.d[l, p] - quicksum(self.t[l, s] for s in ins.active[0, p]))

        # R7
        self.model.addConstrs(self.R[s] == quicksum(
            self.t[l, s] * ins.H[s] * ins.r[l, ins.ids[s]] for l in ins.L) for s in range(ins.S_0))

        # Guardamos la instancia para después
        self.instance = ins

    def run_mip(self):
        '''
        OUTPUT:
            assignment: lista de listas tal que assignment[i] es una lista de abogados 
                        asignados al servicio i
            time_left: diccionario de key (l, p) y value tiempo en horas
                       disponibles del abogado l en el periodo p
            services_rating: lista tal que services_rating[i] es el rating de asignación del
                             servicio i
        '''
        self.model.optimize()

        # Se construyen estructuras a retornar
        ins = self.instance
        assignment = []
        time_left = {}
        services_rating = []

        for l in ins.L:
            for p in range(1, ins.P + 1):
                time_left[l, p] = self.z[l, p].x

        for s in range(ins.S_0):
            a = []
            services_rating.append(self.R[s].x)
            for l in ins.L:
                if self.x[l, s].x:
                    a.append(l)
            assignment.append(a)

        return assignment, time_left, services_rating
