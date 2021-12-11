import numpy as np
import random
from collections import defaultdict, deque
import math
from rating_function import process_ratings, global_register


class Instance:
    '''
    Clase que genera una instancia del modelo
    '''

    def __init__(self, cases, services, lawyers, parents, tmin, base_cases, mode='saa', **kwargs):

        #### ATRIBUTOS COMUNES ####

        # Lista de casos
        self.cases = cases
        # DataFrame de servicios
        self.services = services
        # DataFrame de abogados
        self.lawyers = lawyers
        # Árbol de especializaciones
        self.parents = parents
        # Número de servicios de los casos base
        self.S_0 = 0
        # Horizonte de tiempo
        self.P = 0
        # Diccionario de servicios activos
        self.active = defaultdict(list)
        # Horas semana
        self.h = []
        # Semanas
        self.H = []
        # Abogados
        self.L = []
        # Disponibilidades base de abogados
        self.d = {}
        # Id de cada servicio generado
        self.ids = []
        # Tiempo mínimo a asignar a cualquier abogado
        self.tmin = tmin
        # Lista de casos base
        self.base_cases = []
        # Inicializamos beta
        self.beta = 0

        #### ATRIBUTOS SAA ####
        if mode == 'saa':
            # Número de escenarios
            self.E = kwargs['nscenarios']
            # Tasa de llegada de casos
            self.rate = kwargs['rate']
            # Número de servicios generados
            self.S = 0
            # Periodo de cada servicio
            self.sp = {}
            # Factor de descuento temporal
            self.lambd = kwargs['lambd']
            # Fracción de llegada
            self.frac = 1 - kwargs['arrival']/5

        # Guardamos el modo
        self.mode = mode
        # Se crea la instancia
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

    def generate_service(self, id):
        '''
        Se generan las características del servicio de identificador id
        ------------------------------------------------------------------
        OUTPUT:
            hweeks: horas/semana que requiere el servicio de identificador id
            weeks: horas que requiere el servicio
        '''
        row = self.services[self.services['id'] == id]
        hweeks = random.triangular(
            list(row['minimo hs'])[0], list(row['maximo hs'])[0], list(row['promedio hs'])[0])
        weeks = random.triangular(
            list(row['minimo s'])[0], list(row['maximo s'])[0], list(row['promedio s'])[0])
        return math.ceil(hweeks), math.ceil(weeks)

    def generate_services(self, period, scenario):
        '''
        Se generan las características de los servicios que llegan 
        en un periodo y escenario determinado
        '''
        lambd = self.frac * self.rate
        num = np.random.poisson(lambd)
        for n in range(num):
            p = random.random()
            if p > self.prob:
                continue
            mold = random.choice(self.selected_cases)

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
                    self.active[scenario, per].append(self.S)
                self.S += 1

    def create_instance(self, base_cases):
        '''
        Se genera la instancia a utilizar en el modelo
        '''

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
                    self.active[0, per].append(self.S_0)
                self.S_0 += 1
            self.base_cases.append(c)

        if self.mode == 'saa':
            self.S = self.S_0
            
        # Se dividen los abogados en áreas
        lawyers_to_areas = defaultdict(list)
        for row in self.lawyers.itertuples(index=False):
            for area in row.areas:
                lawyers_to_areas[area].append(row.id)

        # Se pobla lista de abogados a considerar
        if self.mode == 'saa':
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
                        for p in range(1, self.P + 1):
                            if (l, p) not in self.d:
                                self.d[l, p] = list(
                                    self.lawyers[self.lawyers["id"] == l]["hb"])[0]
                        for a in list(self.lawyers[self.lawyers['id'] == l]['areas'])[0]:
                            if a not in areas:
                                queue.append(a)
        elif self.mode == 'greedy':
            for area in lawyers_to_areas:
                if area in base_areas:
                    for l in lawyers_to_areas[area]:
                        if l not in self.L:
                            self.L.append(l)
                            for p in range(1, self.P + 1):
                                if (l, p) not in self.d:
                                    self.d[l, p] = list(
                                        self.lawyers[self.lawyers["id"] == l]["hb"])[0]

        # Se identifican abogados
        lawyers = self.lawyers[self.lawyers['id'].isin(self.L)]

        if self.mode == 'saa':
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

            # Guardamos lista de casos a utilizar
            self.selected_cases = new_cases

            # Generamos todos los escenarios
            for e in range(1, self.E + 1):
                if self.frac:
                    self.generate_services(0, e)
                for p in range(1, self.P + 1):
                    self.generate_services(p, e)

        # Aprovechamos de calcular gamma
        self.gamma = (self.S_0 + 1) * self.beta

        # Calculamos rating
        register = global_register(self.services, self.parents, lawyers)
        self.r = process_ratings(lawyers, list(
            set(self.ids)), self.parents, register, depth=100)

    def update_instance(self, d, base_cases, mode='saa', **kwargs):
        '''
        Función que actualiza instancia
        '''
        #### ATRIBUTOS COMUNES ####

        self.S_0 = 0
        self.P = 0
        self.active = defaultdict(list)
        self.h = []
        self.H = []
        self.L = []
        self.d = d
        self.ids = []
        self.base_cases = []
        self.beta = 0

        #### ATRIBUTOS SAA ####
        if mode == 'saa':
            self.S = 0
            self.sp = {}
            self.frac = 1 - kwargs['arrival']/5
        self.create_instance(base_cases)
