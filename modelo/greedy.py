
from collections import defaultdict


def can_do(instance, time, l, s, div, tmin):
    '''
    Se verifica si abogado l puede realizar servicio s según tiempo
    disponible (time) y tiempo mínimo a asignar (tmin)
    ---------------------------------------------------------------
    INPUT:
        instance: objeto de clase instancia
        l: índice abogado
        s: número de generación del servicio
        time: diccionario de key (l, p) y value tiempo en horas
              disponibles del abogado l en el periodo p
        div: divisor que se aplica al tiempo en horas/semana de un
             servicio
        tmin: tiempo mínimo que se puede asignar a un servicio

    OUTPUT: booleano indicando si es posible que el abogado realice
            el servicio o no
    '''

    for p in range(1, instance.H[s] + 1):
        if time[l, p] < instance.h[s] / div or instance.h[s] / div < tmin:
            return False
    return True


def run_greedy(instance, time_left):
    '''
    OUTPUT:
        assignment: lista de listas tal que assignment[i] es una lista de abogados 
                    asignados al servicio i
        time_left: diccionario de key (l, p) y value tiempo en horas
                   disponibles del abogado l en el periodo p
        time_assigned: diccionario de key (l, p) y value tiempo en horas asignado al abogado 
                       durante el periodo p
        cases_rating: lista tal que cases_rating[j] es el rating de asignación del
                      caso j
    '''
    assignment = []
    cases_rating = []
    time_assigned = defaultdict(int)

    base_cases = instance.base_cases
    for case in base_cases:
        of = 0
        for s in case:
            s_id = instance.ids[s]
            filt = list(
                filter(lambda x: instance.r[x, s_id] > 0, instance.L))
            ordered_lawyers = sorted(
                filt, key=lambda x: instance.r[x, s_id], reverse=True)
            div = 1
            sol = False
            candidates = []
            while instance.h[s] / div >= instance.tmin and div <= len(ordered_lawyers):
                for idx in range(len(ordered_lawyers)):
                    if can_do(instance, time_left, ordered_lawyers[idx], s, div, instance.tmin):
                        candidates.append(ordered_lawyers[idx])
                    if len(candidates) == div:
                        sol = True
                        break
                if sol:
                    break
                div += 1
                candidates = []

            for l in candidates:
                for p in range(1, instance.H[s] + 1):
                    time_left[l, p] -= instance.h[s] / div
                    time_assigned[l, p] += instance.h[s] / div
                of += (instance.h[s] * instance.H[s] * instance.r[l, s_id]) / div
            assignment.append(candidates)
        cases_rating.append(of)

    return assignment, time_left, time_assigned, cases_rating