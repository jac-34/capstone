from model_v2 import load_data, InstanceGenerator
from greedy import GreedyInstanceGenerator, GreedyILModel
import math
import random
import numpy as np
from parameters import *
import pickle
from gurobipy import *


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

class ILModel:
    def __init__(self, instance):

        # Al inicializarse por primera vez se carga la instancia
        # internamente
        self.charge_instance(instance)

    def charge_instance(self, ins):
        ''' 
        Función que carga una instancia y genera todas las restricciones
        y variables del modelo
        '''

        #### MODELO ####
        self.model = Model('Propuesto')

        #### VARIABLES ####
        self.x = self.model.addVars(ins.L, range(ins.S), vtype=GRB.BINARY)
        self.y = self.model.addVars(range(ins.S), vtype=GRB.INTEGER)
        self.t = self.model.addVars(ins.L, range(ins.S), vtype=GRB.CONTINUOUS)
        self.z = self.model.addVars(ins.L, range(ins.E + 1),
                                    range(1, ins.P + 1), vtype=GRB.CONTINUOUS)
        self.n = self.model.addVars(range(ins.S), vtype=GRB.BINARY)
        self.R = self.model.addVars(range(ins.S), vtype=GRB.CONTINUOUS)

        self.model.setParam('OutputFlag', 0)
        self.model.setParam('Presolve', 2)
        self.model.setParam('TimeLimit', 40)
        self.model.update()

        #### FUNCIÓN OBJETIVO ####
        obj = quicksum(self.R[s] - ins.beta * (self.y[s] - 1) - ins.gamma * self.n[s] for s in range(ins.S_0)) + (1/ins.E) * quicksum((ins.lambd ** ins.sp[s]) * (self.R[s] - ins.beta * (self.y[s] - 1) - ins.gamma * self.n[s]) for s in range(ins.S_0, ins.S))

        self.model.setObjective(obj, GRB.MAXIMIZE)

        #### RESTRICCIONES ####

        # R1
        self.model.addConstrs(quicksum(self.t[l, s] for l in ins.L) == ins.h[s] * (1 - self.n[s]) for s in range(ins.S))

        # R2
        self.model.addConstrs(ins.tmin * self.x[l, s] <= self.t[l, s] for s in range(ins.S) for l in ins.L)
        self.model.addConstrs(BIG_M * self.x[l, s] >= self.t[l, s] for s in range(ins.S) for l in ins.L)

        # R3
        for s in range(ins.S):
            for l in ins.L:
                if ins.r[l, ins.ids[s]] == 0:
                    self.model.addConstr(self.x[l, s] == 0)
        
        # R4
        self.model.addConstrs(self.y[s] == quicksum(self.x[l, s] for l in ins.L) for s in range(ins.S))

        # R5
        self.model.addConstrs(1 - self.y[s] <= self.n[s] for s in range(ins.S))
        self.model.addConstrs(self.n[s] <= 1 - self.x[l, s] for s in range(ins.S) for l in ins.L)

        # R6
        for l in ins.L:
            for p in range(1, ins.P + 1):
                self.model.addConstr(self.z[l, 0, p] == ins.d[l, p] - quicksum(self.t[l, s] for s in ins.active[0, p]))
                for e in range(1, ins.E + 1):
                    self.model.addConstr(self.z[l, e, p] == self.z[l, 0, p] - quicksum(self.t[l, s] for s in ins.active[e, p]))
        
        # R7
        self.model.addConstrs(self.R[s] == quicksum(self.t[l, s] * ins.H[s] * ins.r[l, ins.ids[s]] for l in ins.L) for s in range(ins.S))

        # Guardamos la instancia para después
        self.instance = ins

    def run_mip(self):
        '''
        OUTPUT:
            assignment: lista de listas tal que assignment[i] es una lista de abogados 
                        asignados al servicio i
            time_left: diccionario de key (l, p) y value tiempo en horas
                       disponibles del abogado l en el periodo p
            services_rating: lista tal que services_rating[i] es el rating de asignación del
                             servicio i
        '''
        self.model.optimize()

        # Se construyen estructuras a retornar
        ins = self.instance
        assignment = []
        time_left = {}
        services_rating = []

        for l in ins.L:
            for p in range(1, ins.P + 1):
                time_left[l, p] = self.z[l, 0, p].x

        for s in range(ins.S_0):
            a = []
            services_rating.append(self.R[s].x)
            for l in ins.L:
                if self.x[l, s].x:
                    a.append(l)
            assignment.append(a)

        return assignment, time_left, services_rating


if __name__ == "__main__":
    # Cargamos datos
    services, parents, cases, unfiltered_lawyers, specialties_decod, lawyers_decod = load_data()
    ponds = [10]
    for pond in ponds:
        print(f'POND = {pond}')
        # Inicializamos estructuras de datos
        # Modelo
        rp = np.zeros(shape=(REPS, WEEKS))
        nasp = np.zeros(shape=(REPS, WEEKS))
        fssa = np.zeros(shape=(REPS, WEEKS))
        # Greedy
        rp_greedy = np.zeros(shape=(REPS, WEEKS))
        nasp_greedy = np.zeros(shape=(REPS, WEEKS))
        fssa_greedy = np.zeros(shape=(REPS, WEEKS))

        # Fijamos semilla para resultados replicables
        random.seed(7)
        np.random.seed(7)

        for rep in range(REPS):
            print(f'COMENZANDO REP {rep + 1}\n')

            # Día 0 completado
            comp = False
            for w in range(WEEKS):
                print(f'COMENZANDO SEMANA {w + 1}')
                # Servicios que han llegado
                ns = 0

                # Abogados asignados
                # Modelo
                la = 0
                # Greedy
                la_greedy = 0

                # Servicios sin asignar
                # Modelo
                na = 0
                # Greedy
                na_greedy = 0

                # Rating acumulado de servicios
                # Modelo
                r = 0
                # Greedy
                r_greedy = 0

                # Tiempo acumulado servicios
                # Modelo
                t = 0
                # Greedy
                t_greedy = 0

                for arrival in [1, 2, 3, 4, 5]:
                    print(f'Comenzando día {arrival}')

                    # Número de casos a generar
                    ncases = np.random.poisson(RATE / 5)
                    
                    if ncases == 0:
                        if arrival == 5 and comp:
                            d = {}
                            d_greedy = {}
                            for (l, p) in tl:
                                if p >= 2:
                                    d[l, p - 1] = tl[l, p]
                            for (l, p) in tl_greedy:
                                if p >= 2:
                                    d_greedy[l, p - 1] = tl_greedy[l, p]
                            tl = d
                            tl_greedy = d_greedy
                        print('\n')
                        continue

                    # Generamos servicios base
                    base_cases = generate_cases(services, cases, ncases)

                    if not comp:
                        # Creamos las instancias
                        instance = InstanceGenerator(cases, services, unfiltered_lawyers, parents,
                                                    NSCENARIOS, RATE, LAMBDA, T_MIN, pond, base_cases=base_cases, arrival=arrival)
                        instance_greedy = GreedyInstanceGenerator(
                            services, unfiltered_lawyers, parents, T_MIN, pond, base_cases)
                        # Creamos los modelos
                        model = ILModel(instance)
                        model_greedy = GreedyILModel(instance_greedy)
                        comp = True
                    else:
                        # Actualizamos si las instancias y modelos fueron creados anteriormente
                        instance.reboot_instance(tl, base_cases, arrival)
                        instance_greedy.reboot_instance(tl_greedy, base_cases)
                        model.charge_instance(instance)
                        model_greedy.charge_instance(instance_greedy)

                    # Actualizamos servicios que han llegado
                    ns += instance.S_0

                    # Corremos modelos y extraemos info
                    # Modelo
                    print(f'Corriendo modelo propuesto')
                    a, tl, rating = model.run_mip()
                    print(a)
                    for s in range(instance.S_0):
                        if a[s]:
                            for l in a[s]:
                                la += 1
                            r += rating[s]
                            t += instance.h[s] * instance.H[s]
                        else:
                            na += 1

                    # Greedy
                    print(f'Corriendo modelo greedy\n')
                    a_greedy, tl_greedy, rating_greedy = model_greedy.run_mip()
                    print(a_greedy)
                    for s in range(instance.S_0):
                        if a_greedy[s]:
                            for l in a_greedy[s]:
                                la_greedy += 1
                            r_greedy += rating_greedy[s]
                            t_greedy += instance_greedy.h[s] * instance_greedy.H[s]
                        else:
                            na_greedy += 1

                    # Modificamos tl y tl_greedy para la siguiente semana

                    if arrival == 5:
                        d = {}
                        d_greedy = {}
                        for (l, p) in tl:
                            if p >= 2:
                                d[l, p - 1] = tl[l, p]
                        for (l, p) in tl_greedy:
                            if p >= 2:
                                d_greedy[l, p - 1] = tl_greedy[l, p]
                        tl = d
                        tl_greedy = d_greedy

                # Guardamos datos
                if ns:
                    # rating promedio servicios
                    if t:
                        rp[rep, w] = r / t
                    if t_greedy:
                        rp_greedy[rep, w] = r_greedy / t_greedy

                    # número de abogados asignados en promedio
                    nasp[rep, w] = la / ns
                    nasp_greedy[rep, w] = la_greedy / ns

                    # fracción de servicios sin asignar
                    fssa[rep, w] = na / ns
                    fssa_greedy[rep, w] = na_greedy / ns
                else:
                    rp[rep, w] = None
                    rp_greedy[rep, w] = None
                    nasp[rep, w] = None
                    nasp_greedy[rep, w] = None
                    fssa[rep, w] = None
                    fssa_greedy[rep, w] = None

        # Guardamos generados en pickle
        with open(f'resultados/pond{pond}/rp.pickle', 'wb') as file:
            pickle.dump(rp, file)
        with open(f'resultados/pond{pond}/rp_greedy.pickle', 'wb') as file:
            pickle.dump(rp_greedy, file)

        with open(f'resultados/pond{pond}/nasp.pickle', 'wb') as file:
            pickle.dump(nasp, file)
        with open(f'resultados/pond{pond}/nasp_greedy.pickle', 'wb') as file:
            pickle.dump(nasp_greedy, file)

        with open(f'resultados/pond{pond}/fssa.pickle', 'wb') as file:
            pickle.dump(fssa, file)
        with open(f'resultados/pond{pond}/fssa_greedy.pickle', 'wb') as file:
            pickle.dump(fssa_greedy, file)
