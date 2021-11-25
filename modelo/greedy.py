def has_time(s_id, l_id, h, H, time):
    """
    Determina si l_id tiene tiempo para realizar s_id.
    En el caso de ser asi, le resta el tiempo que toma realizarlo
    en el registro time.
    ----------------------------------------------------------------
    INPUT:
        s_id: id del servicio
        l_id: id del abogado
        h: diccionario con key service_id que retorna cuantas hrs semanales
        requiere el service_id
        H: diccionario con key service_id que retorna cuantas semanas requiere
        service_id
        time: diccionario con key (l_id, week) que retorna cuantas hrs disponibles
        tiene el abogado l_id en la semana week
    
    OUTPUT:
        booleano que indica si el abogado tiene tiempo o no.
    """
    for week in range(1, H[s_id] + 1):
        if time[l_id, week] < h[s_id]:
            return False
    # Si estamos aqui, es porque el abogado tiene tiempo para todas las semanas
    # y sera asignado al servicio
    for week in range(1, H[s_id] + 1):
        time[l_id, week] -= h[s_id]
    return True

def greedy(lawyers, S0, s_ids, r, d, h, H):
    """
    Resuelve el problema de asignacion mediante estrategia greedy
    --------------------------------------------------------------
    INPUT:
        lawyers: lista con los ids de los abogados
        case: lista con los ids de los servicios del caso
        r: diccionario con key (l_id, s_id) que retorna el
        rating del abogado l_id para el servicio s_id
        d: diccionario con key l_id que retorna las hrs disponibles
        semanales del abogado l_id
        h: diccionario con key ngen que retorna cuantas hrs semanales
        requiere el servicio ngen
        H: diccionario con key ngen que retorna cuantas semanas requiere
        el servicio ngen

    OUTPUT:
        assignment: diccionario con key ngen que retorna el l_id
        asignado.
        time: diccionario con key (l_id, week) que retorna cuanto
        tiempo restante tiene l_id en la semana week.
        of: objective function, valor de la asignacion realizada
    """
    assignment = {}
    MAX_WEEKS = max(H.values())
    time = {(l_id, week): d[l_id] for l_id in lawyers for week in range(1, MAX_WEEKS + 1)}
    of = 0
    for ngen in S0:
        s_id = s_ids[ngen]
        assignment[ngen] = None
        ordered_lawyers = sorted(lawyers, key=lambda x: r[x, s_id], reverse=True)
        for l_id in ordered_lawyers:
            if has_time(ngen, l_id, h, H, time):
                if r[l_id, s_id] > 0:
                    assignment[ngen] = l_id
                    of += h[ngen] * H[ngen] * r[l_id, s_id]
                    break
    return assignment, time, of
