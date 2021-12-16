import pickle
from simulation import parameters_testing, simulation
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

    # Corremos simulación
    metrics, selection = simulation(services, parents, cases, unfiltered_lawyers, examples_size=2)

    # Guardamos archivos
    with open(f'simulation_results/metrics.pickle', 'wb') as file:
        pickle.dump(metrics, file)
    
    with open(f'simulation_results/selection.pickle', 'wb') as file:
        pickle.dump(selection, file)
    
