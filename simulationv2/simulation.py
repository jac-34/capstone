from heapq import heapify
import math
import random
from collections import defaultdict
import numpy as np
from parameters import *
from gurobipy import *
from model import ILModel
from instance import Instance
from heap import Heap
from time import time


def generate_service(services, id):
    '''
    Se generan las características del servicio de identificador id
    ------------------------------------------------------------------
    OUTPUT:
        hweeks: horas/semana que requiere el servicio de identificador id
        weeks: horas que requiere el servicio
    '''
    row = services[services['id'] == id]
    hweeks = random.triangular(
        list(row['minimo hs'])[0], list(row['maximo hs'])[0], list(row['promedio hs'])[0])
    weeks = random.triangular(
        list(row['minimo s'])[0], list(row['maximo s'])[0], list(row['promedio s'])[0])
    return math.ceil(hweeks), math.ceil(weeks)


def generate_cases(services, cases, ncases):
    '''
    Se generan los casos de un día determinado
    '''
    base_cases = []
    for n in range(ncases):
        mold = random.choice(cases)
        case = {}
        for id in mold:
            hweeks, weeks = generate_service(services, id)
            case[id] = (hweeks, weeks)
        base_cases.append(case)
    return base_cases


def parameters_testing(lambdas, weeks, rates, services, parents, cases, unfiltered_lawyers):
    '''
    Función que genera una simulación de 'weeks' semanas y una repetición.
    Se testean valores de parámetros como lambda y rate.
    ----------------------------------------------------------------------
    OUTPUT:
        assignments, time_lawyers, sa, sr, spr son diccionarios anidados con las siguientes keys:

            * lambd: lambda utilizado
            * rate: rate de llegada de casos utilizado
            * mode: puede ser 'saa' (sample average aproximation) o 'greedy'
            * w: semana (va de 0 en adelante). Si no aparece alguna semana significa que dicha semana no llegaron servicios.

        - assignment[rate]][lamb][mode][w] es una lista donde el elemento i de dicha lista es una lista de los abogados asignados
          al servicio i (respecto a la semana w)
        - time_lawyers[rate][lamb][mode][w] es un diccionario con key l donde el value es el tiempo del abogado en la semana w
          DESPUÉS de la asignación. En este diccionario w va de 1 en adelante, pues se empieza a trabajar desde esa semana.
        - sa[rate][lamb][mode][w] es una lista donde el elemento i es la tupla (h, H) del servicio i
        - sr[rate][lamb][mode][w] es una lista donde el elemento i es el rating del servicio i (respecto a la semana w)
        - spr es lo mismo que sr, pero se consideran los castigos de beta y gamma.

    '''
    # Asignaciones
    assignments = {}
    # Tiempo abogados semana a semana
    time_lawyers = {}
    # Características servicios
    sa = {}
    # Rating servicios
    sr = {}
    # Rating penalizado servicios
    spr = {}
    for rate in rates:
        print(f'COMENZANDO RATE = {rate}\n')
        assignments[rate] = {}
        time_lawyers[rate] = {}
        sa[rate] = {}
        sr[rate] = {}
        spr[rate] = {}
        for lamb in lambdas:
            # Fijamos semilla para resultados replicables
            random.seed(7)
            np.random.seed(7)

            print(f'COMENZANDO LAMBDA = {lamb}')
            assignments[rate][lamb] = {'saa': {}, 'greedy': {}}
            time_lawyers[rate][lamb] = {'saa': {}, 'greedy': {}}
            sa[rate][lamb] = {'saa': {}, 'greedy': {}}
            sr[rate][lamb] = {'saa': {}, 'greedy': {}}
            spr[rate][lamb] = {'saa': {}, 'greedy': {}}
            # Semana 0 completada:
            comp = False
            for w in range(weeks):
                print(f'Comenzando semana {w}')
                # Se determina aleatoriamente en qué día
                # se van a concentrar todos los casos
                arrival = random.choice([1, 2, 3, 4, 5])

                # Número de casos semanales
                ncases = np.random.poisson(rate)
                if ncases == 0:
                    if comp:
                        d_saa = {}
                        d_greedy = {}
                        tla_saa = {}
                        tla_greedy = {}
                        for (l, p) in tl_saa:
                            if p == 1:
                                tla_saa[l] = tl_saa[l, p]
                            else:
                                d_saa[l, p - 1] = tl_saa[l, p]
                        for (l, p) in tl_greedy:
                            if p == 1:
                                tla_greedy[l] = tl_greedy[l, p]
                            else:
                                d_greedy[l, p - 1] = tl_greedy[l, p]
                        tl_saa = d_saa
                        tl_greedy = d_greedy
                        time_lawyers[rate][lamb]['saa'][w + 1] = tla_saa
                        time_lawyers[rate][lamb]['greedy'][w + 1] = tla_greedy
                    print('Ups! No ha llegado ningún servicio\n')
                    continue

                # Generamos servicios base
                base_cases = generate_cases(services, cases, ncases)

                if not comp:
                    instance_saa = Instance(cases, services, unfiltered_lawyers, parents, T_MIN, base_cases, nscenarios=NSCENARIOS,
                                            rate=rate, lambd=lamb, arrival=arrival)
                    instance_greedy = Instance(
                        cases, services, unfiltered_lawyers, parents, T_MIN, base_cases, mode='greedy')
                    m_saa = ILModel(instance_saa)
                    m_greedy = ILModel(instance_greedy)
                    comp = True
                else:
                    instance_saa.update_instance(
                        tl_saa, base_cases, arrival=arrival)
                    instance_greedy.update_instance(
                        tl_greedy, base_cases, mode='greedy')
                    m_saa.charge_instance(instance_saa)
                    m_greedy.charge_instance(instance_greedy)

                # Corremos modelos
                print(f'Corriendo modelo SAA...')
                a_saa, tl_saa, sa_saa, sr_saa, spr_saa = m_saa.run()

                print(f'Corriendo modelo GREEDY...\n')
                a_greedy, tl_greedy, sa_greedy, sr_greedy, spr_greedy = m_greedy.run()

                # Guardamos info
                assignments[rate][lamb]['saa'][w] = a_saa
                assignments[rate][lamb]['greedy'][w] = a_greedy

                sa[rate][lamb]['saa'][w] = sa_saa
                sa[rate][lamb]['greedy'][w] = sa_greedy

                sr[rate][lamb]['saa'][w] = sr_saa
                sr[rate][lamb]['greedy'][w] = sr_greedy

                spr[rate][lamb]['saa'][w] = spr_saa
                spr[rate][lamb]['greedy'][w] = spr_greedy

                # Modificamos los tl's para la próxima semana
                d_saa = {}
                d_greedy = {}
                tla_saa = {}
                tla_greedy = {}
                for (l, p) in tl_saa:
                    if p == 1:
                        tla_saa[l] = tl_saa[l, p]
                    else:
                        d_saa[l, p - 1] = tl_saa[l, p]
                for (l, p) in tl_greedy:
                    if p == 1:
                        tla_greedy[l] = tl_greedy[l, p]
                    else:
                        d_greedy[l, p - 1] = tl_greedy[l, p]
                tl_saa = d_saa
                tl_greedy = d_greedy

                time_lawyers[rate][lamb]['saa'][w + 1] = tla_saa
                time_lawyers[rate][lamb]['greedy'][w + 1] = tla_greedy

    return assignments, time_lawyers, sa, sr, spr

def times_for_next_period(tl, tla):
    '''
    Función que retorna tiempos de abogados y modifica
    array de tiempos de asignación
    '''
    # Tiempos que queda de abogados
    time_lawyers = {}
    for (l, p) in tl:
        if p == 1:
            tla[l] = tl[l, p]
        else:
            time_lawyers[l, p - 1] = tl[l, p]
    return time_lawyers


def simulation(services, parents, cases, unfiltered_lawyers, reps=REPS, weeks=WEEKS, lambd=LAMBDA, tmin=T_MIN, hor=HOR, rate=RATE, nscenarios=NSCENARIOS, seed=7, examples_size = 4):
    '''
    INPUT:
        Los inputs a destacar son 'hor' (horizonte de tiempo simulación interna), 'seed' (semilla para resultados replicables)
        y 'examples_size' que es la cantidad de repeticiones a guardar en las estructuras best_ratings, worst_ratings, best_botados,
        worst_botados y regular_sample
    OUTPUT:
        - metrics: lista tal que metrics[rep] es un diccionario de métricas agregadas de la repetición rep. Las keys y values del 
                   diccionario mencionados son:
            * 'rep': rep
            * 'saa': diccionario que guarda las métricas 'la' (número de abogados asignados), 'ns' (número de servicios generados), 'ra' (rating acumulado),
                     'tta' (diccionario con keys índices de abogados y values el tiempo total asignado)
            * 'greedy': lo mismo que antes, pero para greedy
        - selection: es la lista [best_ratings, worst_ratings, best_botados, worst_botados, regular_sample]
    '''
    # Lista donde se guardan las métricas de cada repetición
    metrics = []
    numlawyers = len(unfiltered_lawyers)

    # Seteamos semilla
    random.seed(seed)
    np.random.seed(seed)

    # Inicializamos heaps
    best_ratings = Heap(lambda t: t[0], max_size=examples_size)
    worst_ratings = Heap(lambda t: -t[0], max_size=examples_size)

    best_botados = Heap(lambda t: -t[0], max_size=examples_size)
    worst_botados = Heap(lambda t: t[0], max_size=examples_size)

    choices = random.sample(range(reps), examples_size)
    regular_sample = []
    t0 = time()

    for rep in range(reps):
        # Inicializamos estructuras de datos a utilizar
        agregated_metric = {'rep': rep, 'saa': {'la': 0, 'ns': 0, 'nb': 0, 'ra': 0, 'tta': 0},
                            'greedy': {'la': 0, 'ns': 0, 'nb': 0, 'ra': 0, 'tta': 0}}
        metric = {'rep': rep, 'saa': {'la': [0] * weeks, 'ns': [0] * weeks, 'nb': [0] * weeks, 'ra': [0] * weeks,
                  'tta': np.zeros(shape=(numlawyers, weeks)), 'tla': np.full(shape=(weeks + 1, numlawyers), fill_value=np.nan),
                  'tda': {w: defaultdict(int) for w in range(weeks)}}, 'greedy': {'la': [0] * weeks, 'ns': [0] * weeks, 'nb': [0] * weeks, 'ra': [0] * weeks,
                  'tta': np.zeros(shape=(numlawyers, weeks)), 'tla': np.full(shape=(weeks + 1, numlawyers), fill_value=np.nan), 'tda': {w: defaultdict(int) for w in range(weeks)}}}
        tr = time()
        print(f'COMENZANDO REPETICIÓN {rep}\n')

        # Ya han llegado servicios al menos una vez
        comp = False
        for w in range(weeks):
            tw = time()
            print(f'COMENZANDO SEMANA {w}')
            # Se determina aleatoriamente en qué día
            # se van a concentrar todos los casos
            arrival = random.choice([1, 2, 3, 4, 5])

            # Número de casos semanales
            ncases = np.random.poisson(rate)
            if ncases == 0:
                if comp:
                    tl_saa = times_for_next_period(tl_saa, metric['saa']['tla'][w + 1])
                    tl_greedy = times_for_next_period(tl_greedy, metric['greedy']['tla'][w + 1])
                print('Ups! No ha llegado ningún servicio\n')
                continue

            # Generamos servicios base
            base_cases = generate_cases(services, cases, ncases)

            if not comp:
                instance_saa = Instance(cases, services, unfiltered_lawyers, parents, tmin, base_cases, nscenarios=nscenarios,
                                        rate=rate, lambd=lambd, arrival=arrival, hor=hor)
                instance_greedy = Instance(cases, services, unfiltered_lawyers, parents, tmin, base_cases, mode='greedy')
                m_saa = ILModel(instance_saa)
                m_greedy = ILModel(instance_greedy)
                comp = True
            else:
                instance_saa.update_instance(tl_saa, base_cases, arrival=arrival)
                instance_greedy.update_instance(tl_greedy, base_cases, mode='greedy')
                m_saa.charge_instance(instance_saa)
                m_greedy.charge_instance(instance_greedy)

            # Corremos modelos
            print(f'Corriendo modelo SAA...')
            tl_saa = m_saa.run(metric['saa'], w)

            print(f'Corriendo modelo GREEDY...')
            tl_greedy = m_greedy.run(metric['greedy'], w)

            # Actualizamos vector de tiempo
            tl_saa = times_for_next_period(tl_saa, metric['saa']['tla'][w + 1])
            tl_greedy = times_for_next_period(tl_greedy, metric['greedy']['tla'][w + 1])
            print(f"Tiempo semana {w}: {round(time() - tw, 3)} s\n")
        # Calculamos valores agregados
        # Para saa:
        agregated_metric['saa']['la'] = sum(metric['saa']['la'])
        agregated_metric['saa']['ns'] = sum(metric['saa']['ns'])
        agregated_metric['saa']['nb'] = sum(metric['saa']['nb'])
        agregated_metric['saa']['ra'] = sum(metric['saa']['ra'])
        agregated_metric['saa']['tta'] = np.sum(metric['saa']['tta'], axis=1)

        # Para greedy
        agregated_metric['greedy']['la'] = sum(metric['greedy']['la'])
        agregated_metric['greedy']['ns'] = sum(metric['greedy']['ns'])
        agregated_metric['greedy']['nb'] = sum(metric['greedy']['nb'])
        agregated_metric['greedy']['ra'] = sum(metric['greedy']['ra'])
        agregated_metric['greedy']['tta'] = np.sum(metric['greedy']['tta'], axis=1)

        # Almacenamos agregated_metric en metrics
        metrics.append(agregated_metric)
        
        # Guardamos repeticiones seleccionadas
        best_ratings.push((agregated_metric['saa']['ra'], metric))
        worst_ratings.push((agregated_metric['saa']['ra'], metric))
        best_botados.push((agregated_metric['saa']['nb'], metric))
        worst_botados.push((agregated_metric['saa']['nb'], metric))

        if rep in choices:
            regular_sample.append(metric)
        
        print(f"Tiempo repeticion {rep}: {round(time() - tr, 3)} s\n")

    best_ratings = [br[1] for br in best_ratings.heap]
    worst_ratings = [wr[1] for wr in worst_ratings.heap]
    best_botados = [bb[1] for bb in best_botados.heap]
    worst_botados = [wb[1] for wb in worst_botados.heap]
    selection = [best_ratings, worst_ratings, best_botados, worst_botados, regular_sample]
    print(f"Tiempo total simulacion: {round(time() - t0, 3)} s")
    return metrics, selection