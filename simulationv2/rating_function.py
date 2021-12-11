import numpy as np
import pandas as pd
import pickle

def compute_level(specialty, parents):
    """
    Calcula el nivel de profundidad de una especialidad o servicio.
    Specialty corresponde al index de la especialidad
    """
    level = 1
    parent = parents[specialty]
    while True:
        if parents[parent] != -1:
            level += 1
            parent = parents[parent]
        else:
            return level

def longest_subbranch(list_of_branches, target):
    """
    Encuentra la rama en comun mas larga entre una lista de 
    ramas y otra rama target
    """
    longest = []

    for branch in list_of_branches:
        if len(branch) != len(target):
            continue
        if len(longest) == len(target):
            return tuple(longest)
        for height in range(len(branch) - len(longest) - 1, -1, -1):
            if branch[height] == target[height]:
                longest.insert(0, target[height])
            else:
                break
    return tuple(longest)

def get_branch(specialty, parents):
    """
    retorna la rama desde la especialidad / servicio hasta el area
    mas amplia y retorna el largo de la rama
    """
    branch = [specialty]
    parent = parents[specialty]
    while parent != -1:
        branch.append(parent)
        parent = parents[parent]
    max_height = len(branch)
    return tuple(branch), max_height

def relative_domain(lawyer, service, parents):
    """
    Retorna a cuantas "generaciones" esta un abogado de un servicio
    en el arbol de especialidades.
    """
    branch, _ = get_branch(service, parents)
    declared_domain = lawyer["declarados"]
    if branch[-1] not in lawyer["areas"]:
        return np.inf, ()
    else:
        lawyers_branches = [get_branch(specialty, parents)[0] for specialty in declared_domain]
        longest = longest_subbranch(lawyers_branches, branch)
        return len(branch) - len(longest), tuple(longest)

def declared_score(lawyer, service, parents, depth=3, tau=0.6):
    """
    Calcula el puntaje de un abogado segun su historial declarado para
    un determinado servicio.

    El parametro depth indica cuanta amplitud se quiere tomar en cuenta
    para el calculo del puntaje

    El parametro tau indica cuanto se ponderan las habilidades declaradas
    niveles mas generales.
    """
    assert 0 < tau < 1, "variable tau debe pertenecer a (0, 1)"
    gens, _ = relative_domain(lawyer, service, parents)
    _, max_height = get_branch(service, parents)
    if gens > depth:
        return 0
    else:
        return tau ** gens

def group_ratings(ratings, max=7, min=1):
    """
    Recibe una lista con notas y retorna un array donde en la entrada
    i-esima se guarda la cantidad de veces que esta la evaluacion i 
    """
    grouping = np.zeros(shape=(max - min + 1))
    for rating in ratings:
        grouping[np.math.ceil()] += 1
    return grouping

def wr(average, quant, global_average, minim=0):
    return (quant / (quant + minim)) * average + (minim / (quant + minim)) * global_average

def global_register(services, parents, lawyers):
    """
    Encuentra los promedios de rating y cuantas veces se trabajo en todas 
    las areas y servicios
    """
    register = {}
    for service_id in services["id"]:
        total_sum = 0
        total_done = 0
        no_history = True
        for i, _ in lawyers.iterrows():
            done, quant, avrge = lawyers.loc[i, ["realizados", "cant", "promedio"]]
            if service_id in done:
                no_history = False
                idx = done.index(service_id)
                total_sum += avrge[idx]
                total_done += quant[idx]
                # total_done += 1      ## Aqui se puede cambiar como se calcula el promedio
        if no_history:
            register[service_id] = None
        else:
            register[service_id] = (total_sum / total_done), total_done
    level = list(services["id"])
    iters = 0
    while level and iters < 8:
        account = {}
        no_history = {}
        for id in level:
            if register[id]:
                no_history[parents[id]] = False
                if parents[id] not in account.keys() and parents[id] != -1:
                    account[parents[id]] = [register[id][0] * register[id][1], register[id][1]]
                else:
                    if parents[id] != -1:
                        account[parents[id]][0] += register[id][0] * register[id][1]
                        account[parents[id]][1] += register[id][1]
        level = []
        for parent in account.keys():
            level.append(parent)
            if not no_history[parent]:
                register[parent] = account[parent][0] / account[parent][1], account[parent][1]
            else:
                register[parent] = None
        iters += 1
    return register

def observed_score(lawyer, service, parents, register, depth=3, tau=0.6, MAX_SCORE=7):
    """
    Calcula el puntaje de un abogado segun su historial observado en
    Impacto Legal para un determinado servicio.

    INPUT:
    lawyer: pd.Series con columnas "realizados", "cant" y "promedio", donde
    cada una de estas columnas contiene listas

    service: int que representa el id de un servicio

    parents: lista donde la entrada i corresponde al padre del servicio con id i

    register: diccionario donde la key corresponde al id de un area/servicio
    y el value es una tupla con el promedio y la cantidad de veces que se ha trabajado 
    en esa area/servicio

    depth: float que señala cuantos padres se quiere considerar en el calculo

    tau: float que señala cuanto se penaliza el puntaje por subir una generacion
    """
    done, quant, avrge = lawyer[["realizados", "cant", "promedio"]]
    branch, max_height = get_branch(service, parents)
    gens, longest = relative_domain(lawyer, service, parents)
    if gens > depth - 1:
        return 0
    else:
        branches = {}
        for d_service, d_quant, d_avrge in zip(done, quant, avrge):
            if d_service == service:
                return wr(d_avrge, d_quant, register[d_service][0]) / MAX_SCORE
            branches[d_service] = get_branch(d_service, parents)
        match = False
        for i in range(1, min([depth, len(branch)])):
            total_sum = 0
            total_quant = 0
            for d_service, d_quant, d_avrge in zip(done, quant, avrge):
                d_branch = branches[d_service]
                if branch[i] in d_branch:
                    match = True
                    total_sum += d_avrge
                    total_quant += d_quant
            if match: 
                sc = wr(total_sum / total_quant, total_quant, register[branch[i]])
                return (sc / MAX_SCORE) * (i ** tau)
        return 0

def rating(lawyer, service, parents, register, alpha=0.5, depth=3, tau=0.6):
    """
    Pondera el puntaje del historial declarado y observado de un abogado
    para un determinado servicio
    """
    OS = observed_score(lawyer, service, parents, register, depth, tau)
    DS = declared_score(lawyer, service, parents, depth, tau=tau)
    return alpha * OS + (1 - alpha) * DS

def process_ratings(lawyers, case, parents, register, alpha=0.5, depth=3, tau=0.6):
    """
    Calcula el rating de cada abogado para cada servicio en un caso.
    Retorna un diccionario con key una tupla (id_lawyer, id_service)
    y con value rating

    INPUT:
    lawyers: pd.DataFrame con la info de los abogados

    case: lista con los id de los servicios que constituyen el caso
    """
    ratings = {}
    for i, _ in lawyers.iterrows():
        lawyer = lawyers.loc[i, :]
        for service in case:
            ratings[(i, service)] = rating(lawyer, service, parents, register, alpha, depth, tau)
    return ratings

def save_ratings(lawyers, services, parents, register, lawyer_decod, specialty_decod):
    r = process_ratings(lawyers, services, parents, register)
    service_decod = {key: value for key, value in specialty_decod.items() if key in services}
    df = pd.DataFrame(columns=lawyers_decod.values(), index=service_decod.values())
    for i, _ in lawyers.iterrows():
        lawyer = lawyers.loc[i, :]
        for service in services:
            df.loc[service_decod[service], lawyer_decod[i]] = round(rating(lawyer, service, parents, register), 3)
    df.to_excel("ratings.xlsx")


if __name__ == "__main__":
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

    file = open('decodificacion.pickle', 'rb')
    services_decod = pickle.load(file)
    file.close()

    file = open('decod_nombres.pickle', 'rb')
    lawyers_decod = pickle.load(file)
    file.close()

    register = global_register(servicios, padres, abogados)
    case_idx = 0
    r = process_ratings(abogados, casos[case_idx], padres, register)
    for lawyer, service in r:
        if r[lawyer, service] != 0:
            print(f"r_[{lawyer}, {service}] = {r[lawyer, service]}")
        else: print(f"r_[{lawyer}, {service}] = {r[lawyer, service]}")

    ### Guardar puntajes de todos los abogados para todos los servicios ###
    save_ratings(abogados, list(servicios["id"]), padres, register, lawyers_decod, services_decod)
    print(lawyers_decod)
    