import numpy as np
import random
from collections import defaultdict, deque
import math
from rating_function import process_ratings, global_register


#### FUNCIONES ####


def generate_service(id, services, parents):
    '''
    Recibe el id de un servicio junto al DataFrame de servicios
    y la lista de padres

    Retorna horas/semana, semanas y área del servicio en cuestión
    '''
    row = services[services['id'] == id]
    hweeks = random.triangular(
        list(row['minimo hs'])[0], list(row['maximo hs'])[0], list(row['promedio hs'])[0])
    weeks = random.triangular(
        list(row['minimo s'])[0], list(row['maximo s'])[0], list(row['promedio s'])[0])
    return math.ceil(hweeks), math.ceil(weeks)


class InstanceGenerator:
    '''
    Clase que genera una instancia del modelo
    '''

    def __init__(self, cases, services, lawyers, parents, nscenarios, rate, lambd, tmin, base_case=None):
        # Lista de casos
        self.cases = cases

        # DataFrame de servicios
        self.services = services

        # DataFrame de abogados
        self.lawyers = lawyers

        # Árbol de especializaciones
        self.parents = parents

        # Número de escenarios
        self.E = nscenarios

        # Tasa de llegada de casos
        self.rate = rate

        # Número de servicios generados
        self.S = 0

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

        # Periodo de cada servicio
        self.sp = {}

        # Otros parámetros
        self.lambd = lambd
        self.tmin = tmin

        self.create_instance(base_case)

    def service_area(self, id):
        '''
        Recibe id de un servicio y la lista de padres

        Retorna el área del servicio 
        '''
        current = id
        while self.parents[current] != -1:
            current = self.parents[current]
        return current

    def generate_service(self, id):
        '''
        Recibe el id de un servicio junto al DataFrame de servicios
        y la lista de padres

        Retorna horas/semana, semanas y área del servicio en cuestión
        '''
        row = self.services[self.services['id'] == id]
        hweeks = random.triangular(
            list(row['minimo hs'])[0], list(row['maximo hs'])[0], list(row['promedio hs'])[0])
        weeks = random.triangular(
            list(row['minimo s'])[0], list(row['maximo s'])[0], list(row['promedio s'])[0])
        return math.ceil(hweeks), math.ceil(weeks)

    def generate_services(self, period, scenario, frac=1):
        '''
        Recibe el periodo de llegada y el escenario en cuestión

        Agrega los servicios generados a self.activos
        '''
        num = np.random.poisson(frac * self.rate)
        for n in range(num):
            p = random.random()
            if p > self.prob:
                continue
            mold = random.choice(self.cases)

            for id in mold:
                hweeks, weeks = self.generate_service(id)
                self.h.append(hweeks)
                self.H.append(weeks)
                self.ids.append(id)
                self.sp[self.S] = period
                if hweeks * weeks > self.beta:
                    self.beta = hweeks * weeks
                lim_sup = min(self.P, period + weeks)
                for per in range(period + 1, lim_sup + 1):
                    self.active[(scenario, per)].append(self.S)
                self.S += 1

    def create_instance(self, base_case):
        '''
        Se genera una instancia del modelo
        '''
        # Generación caso base
        if base_case:
            mold = base_case
        else:
            mold = random.choice(self.cases)
        arrival = random.choice([1, 2, 3, 4, 5])

        frac = None
        if arrival != 5:
            frac = 1 - arrival/5

        # Creación servicios base
        base_areas = set()

        # Aprovechamos de calcular el parámetro beta
        self.beta = 0
        for id in mold:
            hweeks, weeks = self.generate_service(id)
            area = self.service_area(id)
            base_areas.add(area)
            self.h.append(hweeks)
            self.H.append(weeks)
            self.ids.append(id)
            if weeks > self.P:
                self.P = weeks
            if hweeks * weeks > self.beta:
                self.beta = hweeks * weeks
            for per in range(1, weeks + 1):
                self.active[(0, per)].append(self.S)
            self.S += 1
            self.S_0 += 1

        # Se dividen los abogados en áreas
        lawyers_to_areas = defaultdict(list)
        for row in self.lawyers.itertuples(index=False):
            for area in row.areas:
                lawyers_to_areas[area].append(row.id)

        # Se pobla lista de abogados a considerar
        areas = []
        queue = deque(base_areas)
        while len(queue) > 0:
            area = queue.popleft()
            if area in areas:
                continue
            areas.append(area)
            for l in lawyers_to_areas[area]:
                if l not in self.L:
                    self.L.append(l)
                    # Poblamos d
                    self.d[l] = list(
                        self.lawyers[self.lawyers["id"] == l]["hb"])[0]
                    for a in list(self.lawyers[self.lawyers['id'] == l]['areas'])[0]:
                        if a not in areas:
                            queue.append(a)

        # Se actualiza tabla de abogados
        self.lawyers = self.lawyers[self.lawyers['id'].isin(self.L)]

        # Se filtran casos a utilizar
        new_cases = []
        for case in self.cases:
            new = []
            for s in case:
                if self.service_area(s) in areas:
                    new.append(s)
            if new:
                new_cases.append(new)

        # Probabilidad de elegir algún caso
        self.prob = len(new_cases) / len(self.cases)

        # Actualizamos la lista de casos
        self.cases = new_cases

        # Generamos todos los escenarios
        for e in range(1, self.E + 1):
            if frac:
                self.generate_services(0, e, frac=frac)
            for p in range(1, self.P + 1):
                self.generate_services(p, e)

        # Aprovechamos de calcular gamma
        self.gamma = (self.S + 1) * self.beta

        # Calculamos rating
        register = global_register(self.services, self.parents, self.lawyers)
        self.r = process_ratings(self.lawyers, list(
            set(self.ids)), self.parents, register, depth=100)