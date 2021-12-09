from parameters import *
from gurobipy import *


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
        self.model = Model()

        if ins.mode == 'saa':
            S = ins.S
        else:
            S = ins.S_0

        #### VARIABLES ####
        self.x = self.model.addVars(ins.L, range(S), vtype=GRB.BINARY)
        self.y = self.model.addVars(range(S), vtype=GRB.INTEGER)
        self.t = self.model.addVars(ins.L, range(S), vtype=GRB.CONTINUOUS)
        if ins.mode == 'saa':
            self.z = self.model.addVars(ins.L, range(ins.E + 1),
                                        range(1, ins.P + 1), vtype=GRB.CONTINUOUS)
        else:
            self.z = self.model.addVars(ins.L, range(
                1, ins.P + 1), vtype=GRB.CONTINUOUS)
        self.n = self.model.addVars(range(S), vtype=GRB.BINARY)
        self.R = self.model.addVars(range(S), vtype=GRB.CONTINUOUS)

        #### HIPERPARÁMETROS DE GUROBI ####
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('Presolve', 2)
        self.model.setParam('TimeLimit', 300)
        self.model.update()

        #### FUNCIÓN OBJETIVO ####
        if ins.mode == 'saa':
            obj = quicksum(self.R[s] - ins.beta * (self.y[s] - 1) -
                           ins.gamma * self.n[s] for s in range(ins.S_0)) + (1/ins.E) * quicksum((ins.lambd ** ins.sp[s]) * (self.R[s] -
                                                                                                                             ins.beta * (self.y[s] + self.n[s] - 1)) for s in range(ins.S_0, ins.S))
        else:
            obj = quicksum(self.R[s] - ins.beta * (self.y[s] - 1) -
                           ins.gamma * self.n[s] for s in range(ins.S_0))

        self.model.setObjective(obj, GRB.MAXIMIZE)

        #### RESTRICCIONES ####

        # R1
        self.model.addConstrs(quicksum(
            self.t[l, s] for l in ins.L) == ins.h[s] * (1 - self.n[s]) for s in range(S))

        # R2
        self.model.addConstrs(
            ins.tmin * self.x[l, s] <= self.t[l, s] for s in range(S) for l in ins.L)
        self.model.addConstrs(
            BIG_M * self.x[l, s] >= self.t[l, s] for s in range(S) for l in ins.L)

        # R3
        for s in range(S):
            for l in ins.L:
                if ins.r[l, ins.ids[s]] == 0:
                    self.model.addConstr(self.x[l, s] == 0)

        # R4
        self.model.addConstrs(self.y[s] == quicksum(
            self.x[l, s] for l in ins.L) for s in range(S))

        # R5
        self.model.addConstrs(1 - self.y[s] <= self.n[s] for s in range(S))
        self.model.addConstrs(self.n[s] <= 1 - self.x[l, s]
                              for s in range(S) for l in ins.L)

        # R6
        if ins.mode == 'saa':
            for l in ins.L:
                for p in range(1, ins.P + 1):
                    self.model.addConstr(self.z[l, 0, p] == ins.d[l, p] - quicksum(self.t[l, s] for s in ins.active[0, p]))
                    for e in range(1, ins.E + 1):
                        self.model.addConstr(self.z[l, e, p] == self.z[l, 0, p] - quicksum(self.t[l, s] for s in ins.active[e, p]))
        else:
            for l in ins.L:
                for p in range(1, ins.P + 1):
                    self.model.addConstr(self.z[l, p] == ins.d[l, p] - quicksum(self.t[l, s] for s in ins.active[0, p]))

        # R7
        self.model.addConstrs(self.R[s] == quicksum(self.t[l, s] * ins.H[s] * ins.r[l, ins.ids[s]] for l in ins.L) for s in range(S))

        # Guardamos la instancia para después
        self.instance = ins

    def run(self):
        '''
        OUTPUT:
            assignment: lista de listas tal que assignment[i] es una lista de abogados 
                        asignados al servicio i
            time_left: diccionario de key (l, p) y value tiempo en horas
                       disponibles del abogado l en el periodo p
            sr: lista tal que sr[i] es el rating de asignación del
                servicio i
            spr: lista tal que spr[i] es el rating penalizado de asignación
                 del servicio i.
        '''
        self.model.optimize()

        # Se construyen estructuras a retornar
        ins = self.instance
        # Asignaciones
        assignment = []
        # Tiempo abogados
        time_left = {}
        # Rating de servicios
        sr = []
        # Rating de servicios penalizado
        spr = []

        if ins.mode == 'saa':
            for l in ins.L:
                for p in range(1, ins.P + 1):
                    time_left[l, p] = self.z[l, 0, p].x
        else:
            for l in ins.L:
                for p in range(1, ins.P + 1):
                    time_left[l, p] = self.z[l, p].x

        for s in range(ins.S_0):
            a = []
            sr.append(self.R[s].x)
            spr.append(self.R[s].x - ins.beta *
                       (self.y[s].x - 1) - ins.gamma * self.n[s].x)
            for l in ins.L:
                if self.x[l, s].x:
                    a.append(l)
            assignment.append(a)

        return assignment, time_left, sr, spr
