import numpy as np
import pandas as pd
import random
from collections import defaultdict, deque
import pickle
import math

## FUNCIONES ÚTILES ##


def generar_caracteristicas(id, servicios, padres):
    '''
    Recibe el id de un servicio junto al dataframe y
    retorna caracteristicas del servicio
    '''
    row = servicios[servicios['id'] == id]
    hsemanas = random.triangular(
        list(row['minimo hs'])[0], list(row['maximo hs'])[0], list(row['promedio hs'])[0])
    semanas = random.triangular(
        list(row['minimo s'])[0], list(row['maximo s'])[0], list(row['promedio s'])[0])
    area = area_servicio(id, padres)
    return math.ceil(hsemanas), math.ceil(semanas), area


def area_servicio(id, padres):
    actual = id
    while padres[actual] != -1:
        actual = padres[actual]
    return actual


class Servicio:

    def __init__(self, id, ngen, servicios, padres, escenario, periodo):
        self.id = id
        self.ngen = ngen
        self.hsemanas, self.semanas, self.area = generar_caracteristicas(
            id, servicios, padres)
        self.escenario = escenario
        self.periodo = periodo


class GeneradorInstancia:

    def __init__(self, casos, servicios, abogados, padres, N=1000, tasa=1.25):
        self.lista_casos = casos
        self.tabla_servicios = servicios
        self.tabla_abogados = abogados
        self.tasa = tasa  # tasa de llegada de casos
        self.n_escenarios = N
        self.padres = padres  # árbol de especialidades
        self.servicios = []  # lista de objetos Servicio
        self.activos = {}  # diccionario que almacena servicios activos
        self.generados = 0  # cantidad de servicios generados
        self.fraccion = None  # fracción de tiempo entre periodo 0 y resto de semana

    def inicializar_generador(self):

        # Generación de servicios base
        molde = random.choice(self.lista_casos)
        llegada = random.choice([1, 2, 3, 4, 5])  # días de la semana
        if llegada != 5:
            self.fraccion = 1 - llegada/5

        horizonte = 0
        areas_base = set()
        for id in molde:
            servicio = Servicio(id, self.generados,
                                self.tabla_servicios, self.padres,  0, -1)
            areas_base.add(servicio.area)
            self.generados += 1

            if servicio.semanas > horizonte:
                horizonte = servicio.semanas

            self.servicios.append(servicio)

        # Se agregan servicios base en diccionario de servicios activos
        self.horizonte = horizonte
        self.activos = {(e, p): [] for e in range(0, self.n_escenarios + 1)
                        for p in range(1, self.horizonte + 1)}
        for s in self.servicios:
            for p in range(1, s.semanas + 1):
                self.activos[(0, p)].append(s.ngen)

        # Se dividen a los abogados en áreas
        abogados_segun_areas = defaultdict(list)
        for row in self.tabla_abogados.itertuples(index=False):
            for area in row.areas:
                abogados_segun_areas[area].append(row.id)

        # Se crea lista de áreas a considerar
        abogados_agregados = []
        self.areas = []
        cola = deque(areas_base)
        while len(cola) > 0:
            area = cola.popleft()
            if area in self.areas:
                continue
            self.areas.append(area)
            for id_abogado in abogados_segun_areas[area]:
                if id_abogado not in abogados_agregados:
                    abogados_agregados.append(id_abogado)
                    for a in list(self.tabla_abogados[self.tabla_abogados['id'] == id_abogado]['areas'])[0]:
                        if a not in self.areas:
                            cola.append(a)

        # Se actualiza la tabla de abogados
        self.tabla_abogados = self.tabla_abogados[self.tabla_abogados['id'].isin(
            abogados_agregados)]

        # Filtrar casos a utilizar y calcular probabilidades
        nuevos_casos = []

        for caso in self.lista_casos:
            nuevo = []
            for s in caso:
                if area_servicio(s, self.padres) in self.areas:
                    nuevo.append(s)
            if nuevo:
                nuevos_casos.append(nuevo)
        C = len(self.lista_casos)
        C1 = len(nuevos_casos)

        # Probabilidad de elegir algun caso
        self.prob_caso = C1/C

        # Actualizamos casos
        self.lista_casos = nuevos_casos

        # Generamos todos los escenarios
        for escenario in range(1, self.n_escenarios + 1):
            if self.fraccion:
                self.generar_servicios(0, escenario, fraccion=self.fraccion)
            for periodo in range(1, self.horizonte + 1):
                self.generar_servicios(periodo, escenario)

        return self.servicios, self.activos, self.tabla_abogados

    def generar_servicios(self, periodo, escenario, fraccion=1):
        num_casos = np.random.poisson(fraccion * self.tasa)
        for n in range(num_casos):
            p = random.random()
            if p > self.prob_caso:
                continue
            molde = random.choice(self.lista_casos)

            for id in molde:
                servicio = Servicio(
                    id, self.generados, self.tabla_servicios, self.padres, escenario, periodo)
                self.generados += 1
                self.servicios.append(servicio)
                for per in range(periodo + 1, self.horizonte + 1):
                    self.activos[(escenario, per)].append(servicio.ngen)



# Cargamos objetos


file = open('servicios.pickle', 'rb')
servicios = pickle.load(file)
file.close()

file = open('padres.pickle', 'rb')
padres = pickle.load(file)
file.close()

file = open('casos.pickle', 'rb')
casos = pickle.load(file)
file.close()

file = open('abogados.pickle', 'rb')
abogados = pickle.load(file)
file.close()

instancia = GeneradorInstancia(casos, servicios, abogados, padres)

servicios, activos, tabla = instancia.inicializar_generador()
