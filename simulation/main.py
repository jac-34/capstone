import pickle
from simulation import parameters_testing
from parameters import *


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


if __name__ == "__main__":
    # Cargamos datos
    services, parents, cases, unfiltered_lawyers, specialties_decod, lawyers_decod = load_data()

    # Definimos lista de lambdas y rates a testear
    lambdas = [0.2, 0.4, 0.6, 0.8, 0.9]
    rates = [1.25, 3, 4]

    assignments, time_lawyers, sr, spr = parameters_testing(
        lambdas, WEEKS, rates, services, parents, cases, unfiltered_lawyers)

    '''
    assignments, time_lawyers, sr, spr son diccionarios anidados con las siguientes keys:

        * lambd: lambda utilizado
        * rate: rate de llegada de casos utilizado
        * mode: puede ser 'saa' (sample average aproximation) o 'greedy'
        * w: semana (va de 0 en adelante). Si no aparece alguna semana significa que dicha semana no llegaron servicios.

    - assignments[lamb][rate][mode][w] es una lista donde el elemento i de dicha lista es una lista de los abogados asignados
      al servicio i (respecto a la semana w)
    - time_lawyers[lamb][rate][mode][w] es un diccionario con key (l, p) donde el value es la cantidad de tiempo disponible del abogado
      l en el periodo p (la semana w corresponde al periodo 0).
    - sr[lamb][rate][mode][w] es una lista donde el elemento i es el rating del servicio i (respecto a la semana w)
    - spr es lo mismo que sr, pero se consideran los castigos de beta y gamma.
    '''

    # Guardamos archivos
    with open(f'results/assignments.pickle', 'wb') as file:
        pickle.dump(assignments, file)

    with open(f'results/assignments.pickle', 'wb') as file:
        pickle.dump(time_lawyers, file)

    with open(f'results/assignments.pickle', 'wb') as file:
        pickle.dump(sr, file)

    with open(f'results/assignments.pickle', 'wb') as file:
        pickle.dump(spr, file)
