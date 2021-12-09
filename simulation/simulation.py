import math
import random
import numpy as np
from parameters import *
from gurobipy import *
from model import ILModel
from instance import Instance


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
    # Asignaciones
    assignments = {}
    # Tiempo abogados semana a semana
    time_lawyers = {}
    # Rating servicios
    sr = {}
    # Rating penalizado servicios
    spr = {}
    for rate in rates:
        print(f'COMENZANDO RATE = {rate}\n')
        assignments[rate] = {}
        time_lawyers[rate] = {}
        sr[rate] = {}
        spr[rate] = {}
        for lamb in lambdas:
            # Fijamos semilla para resultados replicables
            random.seed(7)
            np.random.seed(7)

            print(f'COMENZANDO LAMBDA = {lamb}')
            assignments[rate][lamb] = {'saa': {}, 'greedy': {}}
            time_lawyers[rate][lamb] = {'saa': {}, 'greedy': {}}
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
                        for (l, p) in tl_saa:
                            if p >= 2:
                                d_saa[l, p - 1] = tl_saa[l, p]
                        for (l, p) in tl_greedy:
                            if p >= 2:
                                d_greedy[l, p - 1] = tl_greedy[l, p]
                        tl_saa = d_saa
                        tl_greedy = d_greedy
                    print('Ups! No ha llegado ningún servicio\n')
                    continue

                # Generamos servicios base
                base_cases = generate_cases(services, cases, ncases)

                if not comp:
                    instance_saa = Instance(cases, services, unfiltered_lawyers, parents, T_MIN, base_cases, nscenarios=NSCENARIOS,
                                            rate=rate, lambd=lamb, arrival=arrival)
                    instance_greedy = Instance(cases, services, unfiltered_lawyers, parents, T_MIN, base_cases, mode='greedy')
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
                a_saa, tl_saa, sr_saa, spr_saa = m_saa.run()

                print(f'Corriendo modelo GREEDY...\n')
                a_greedy, tl_greedy, sr_greedy, spr_greedy = m_greedy.run()

                # Guardamos info
                assignments[rate][lamb]['saa'][w] = a_saa
                assignments[rate][lamb]['greedy'][w] = a_greedy

                time_lawyers[rate][lamb]['saa'][w] = tl_saa
                time_lawyers[rate][lamb]['greedy'][w] = tl_greedy

                sr[rate][lamb]['saa'][w] = sr_saa
                sr[rate][lamb]['greedy'][w] = sr_greedy
                
                spr[rate][lamb]['saa'][w] = spr_saa
                spr[rate][lamb]['greedy'][w] = spr_greedy

                # Modificamos los tl's para la próxima semana
                d_saa = {}
                d_greedy = {}
                for (l, p) in tl_saa:
                    if p >= 2:
                        d_saa[l, p - 1] = tl_saa[l, p]
                for (l, p) in tl_greedy:
                    if p >= 2:
                        d_greedy[l, p - 1] = tl_greedy[l, p]
                tl_saa = d_saa
                tl_greedy = d_greedy
    
    return assignments, time_lawyers, sr, spr



                    








# if __name__ == "__main__":
#     # Cargamos datos
#     services, parents, cases, unfiltered_lawyers, specialties_decod, lawyers_decod = load_data()
#     ponds = [10]
#     for pond in ponds:
#         print(f'POND = {pond}')
#         # Inicializamos estructuras de datos
#         # Modelo
#         rp = np.zeros(shape=(REPS, WEEKS))
#         ra = np.zeros(shape=(REPS, WEEKS))
#         rac = np.zeros(shape=(REPS, WEEKS))
#         nasp = np.zeros(shape=(REPS, WEEKS))
#         fssa = np.zeros(shape=(REPS, WEEKS))
#         # Greedy
#         rp_greedy = np.zeros(shape=(REPS, WEEKS))
#         ra_greedy = np.zeros(shape=(REPS, WEEKS))
#         rac_greedy = np.zeros(shape=(REPS, WEEKS))
#         nasp_greedy = np.zeros(shape=(REPS, WEEKS))
#         fssa_greedy = np.zeros(shape=(REPS, WEEKS))



#         # Fijamos semilla para resultados replicables
#         random.seed(7)
#         np.random.seed(7)

#         for rep in range(REPS):
#             print(f'COMENZANDO REP {rep + 1}\n')

#             # Día 0 completado
#             comp = False
#             for w in range(WEEKS):
#                 print(f'COMENZANDO SEMANA {w + 1}')
#                 # Servicios que han llegado
#                 ns = 0

#                 # Abogados asignados
#                 # Modelo
#                 la = 0
#                 # Greedy
#                 la_greedy = 0

#                 # Servicios sin asignar
#                 # Modelo
#                 na = 0
#                 # Greedy
#                 na_greedy = 0

#                 # Rating acumulado de servicios
#                 # Modelo
#                 r = 0
#                 # Greedy
#                 r_greedy = 0

#                 # Tiempo acumulado servicios
#                 # Modelo
#                 t = 0
#                 # Greedy
#                 t_greedy = 0


#                 for arrival in [1, 2, 3, 4, 5]:
#                     print(f'Comenzando día {arrival}')

#                     # Número de casos a generar
#                     ncases = np.random.poisson(RATE / 5)
                    
#                     if ncases == 0:
#                         if arrival == 5 and comp:
#                             d = {}
#                             d_greedy = {}
#                             for (l, p) in tl:
#                                 if p >= 2:
#                                     d[l, p - 1] = tl[l, p]
#                             for (l, p) in tl_greedy:
#                                 if p >= 2:
#                                     d_greedy[l, p - 1] = tl_greedy[l, p]
#                             tl = d
#                             tl_greedy = d_greedy
#                         continue

#                     # Generamos servicios base
#                     base_cases = generate_cases(services, cases, ncases)

#                     if not comp:
#                         # Creamos las instancias
#                         instance = InstanceGenerator(cases, services, unfiltered_lawyers, parents,
#                                                     NSCENARIOS, RATE, LAMBDA, T_MIN, pond, base_cases=base_cases, arrival=arrival)
#                         instance_greedy = GreedyInstanceGenerator(
#                             services, unfiltered_lawyers, parents, T_MIN, pond, base_cases)
#                         # Creamos los modelos
#                         model = ILModel(instance)
#                         model_greedy = GreedyILModel(instance_greedy)
#                         comp = True
#                     else:
#                         # Actualizamos si las instancias y modelos fueron creados anteriormente
#                         instance.reboot_instance(tl, base_cases, arrival)
#                         instance_greedy.reboot_instance(tl_greedy, base_cases)
#                         model.charge_instance(instance)
#                         model_greedy.charge_instance(instance_greedy)

#                     # Actualizamos servicios que han llegado
#                     ns += instance.S_0

#                     # Corremos modelos y extraemos info
#                     # Modelo
#                     print(f'Corriendo modelo propuesto')
#                     a, tl, rating = model.run_mip()
#                     print(a)
#                     for s in range(instance.S_0):
#                         if a[s]:
#                             for l in a[s]:
#                                 la += 1
#                             r += rating[s]
#                             ra[rep, w] += rating[s]
#                             rac[rep, w] += rating[s] - instance.beta * (model.y[s].x - 1) - instance.gamma * model.n[s].x
#                             t += instance.h[s] * instance.H[s]
#                         else:
#                             na += 1
#                     print('Rating acumulado\n')
#                     print(ra[rep, w])

#                     print('Rating acumulado castigado\n')
#                     print(ra_greedy[rep, w])

#                     # Greedy
#                     print(f'Corriendo modelo greedy\n')
#                     a_greedy, tl_greedy, rating_greedy = model_greedy.run_mip()
#                     print(a_greedy)
#                     for s in range(instance.S_0):
#                         if a_greedy[s]:
#                             for l in a_greedy[s]:
#                                 la_greedy += 1
#                             r_greedy += rating_greedy[s]
#                             ra_greedy[rep, w] += rating[s]
#                             rac_greedy[rep, w] += rating[s] - instance.beta * (model.y[s].x - 1) - instance.gamma * model.n[s].x
#                             t_greedy += instance_greedy.h[s] * instance_greedy.H[s]
#                         else:
#                             na_greedy += 1
                            
#                     print('Rating acumulado\n')
#                     print(ra_greedy[rep, w])

#                     print('Rating acumulado castigado\n')
#                     print(rac_greedy[rep, w])

#                     # Modificamos tl y tl_greedy para la siguiente semana

#                     if arrival == 5:
#                         d = {}
#                         d_greedy = {}
#                         for (l, p) in tl:
#                             if p >= 2:
#                                 d[l, p - 1] = tl[l, p]
#                         for (l, p) in tl_greedy:
#                             if p >= 2:
#                                 d_greedy[l, p - 1] = tl_greedy[l, p]
#                         tl = d
#                         tl_greedy = d_greedy

#                 # Guardamos datos
#                 if ns:
#                     # rating promedio servicios
#                     if t:
#                         rp[rep, w] = r / t
#                     if t_greedy:
#                         rp_greedy[rep, w] = r_greedy / t_greedy

#                     # número de abogados asignados en promedio
#                     nasp[rep, w] = la / ns
#                     nasp_greedy[rep, w] = la_greedy / ns

#                     # fracción de servicios sin asignar
#                     fssa[rep, w] = na / ns
#                     fssa_greedy[rep, w] = na_greedy / ns
#                 else:
#                     rp[rep, w] = None
#                     rp_greedy[rep, w] = None
#                     nasp[rep, w] = None
#                     nasp_greedy[rep, w] = None
#                     fssa[rep, w] = None
#                     fssa_greedy[rep, w] = None

#         # Guardamos generados en pickle
#         with open(f'resultados/pond{pond}/rp.pickle', 'wb') as file:
#             pickle.dump(rp, file)
#         with open(f'resultados/pond{pond}/rp_greedy.pickle', 'wb') as file:
#             pickle.dump(rp_greedy, file)

#         with open(f'resultados/pond{pond}/nasp.pickle', 'wb') as file:
#             pickle.dump(nasp, file)
#         with open(f'resultados/pond{pond}/nasp_greedy.pickle', 'wb') as file:
#             pickle.dump(nasp_greedy, file)

#         with open(f'resultados/pond{pond}/fssa.pickle', 'wb') as file:
#             pickle.dump(fssa, file)
#         with open(f'resultados/pond{pond}/fssa_greedy.pickle', 'wb') as file:
#             pickle.dump(fssa_greedy, file)

#         with open(f'resultados/pond{pond}/ra.pickle', 'wb') as file:
#             pickle.dump(fssa, file)
#         with open(f'resultados/pond{pond}/ra_greedy.pickle', 'wb') as file:
#             pickle.dump(fssa_greedy, file)

#         with open(f'resultados/pond{pond}/rac.pickle', 'wb') as file:
#             pickle.dump(fssa, file)
#         with open(f'resultados/pond{pond}/rac_greedy.pickle', 'wb') as file:
#             pickle.dump(fssa_greedy, file)
