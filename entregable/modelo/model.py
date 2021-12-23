from parameters import *
from pyomo.environ import *
from pyomo.opt import SolverFactory


class ILModel:
    '''
    Clase que crea el modelo
    '''
    def __init__(self, instance):

        # Al inicializarse por primera vez se carga la instancia
        # internamente
        self.charge_instance(instance)

    def charge_instance(self, ins):
        ''' 
        Función que carga una instancia y genera todas las restricciones
        y variables del modelo
        ---------------------------------------------------------------
        INPUT:
            ins: instancia de clase Instance
        '''
        #### MODELO ####
        self.model = ConcreteModel()

        if ins.mode == 'saa':
            S = ins.S
        else:
            S = ins.S_0

        #### VARIABLES ####
        self.model.x = Var(ins.L, range(S), domain=Binary)
        self.model.y = Var(range(S), domain=NonNegativeReals)
        self.model.t = Var(ins.L, range(S), domain=NonNegativeReals)
        self.model.u = Var(ins.L, range(1, ins.P + 1), domain=NonNegativeReals)
        if ins.mode == 'saa':
            self.model.z = Var(ins.L, range(1, ins.E + 1),
                               range(1, ins.T + 1), domain=NonNegativeReals)
        self.model.n = Var(range(S), domain=Binary)
        self.model.R = Var(range(S), domain=NonNegativeReals)

        #### FUNCIÓN OBJETIVO ####
        if ins.mode == 'saa':
            self.model.obj = Objective(expr=sum(self.model.R[s] - ins.beta * (self.model.y[s] - 1) - ins.gamma * self.model.n[s] for s in range(ins.S_0)) +
                                       (1/ins.E) * sum((ins.lambd ** ins.sp[s]) * (self.model.R[s] - ins.beta * (self.model.y[s] - 1 + self.model.n[s]))
                                       for s in range(ins.S_0, ins.S)), sense=maximize)
        else:
            self.model.obj = Objective(expr=sum(self.model.R[s] - ins.beta * (self.model.y[s] - 1) - ins.gamma * self.model.n[s] for s in range(ins.S_0)), sense=maximize)
                                       
        #### RESTRICCIONES ####
        
        self.model.r1 = ConstraintList()
        self.model.r2 = ConstraintList()
        self.model.r3 = ConstraintList()
        self.model.r4 = ConstraintList()
        self.model.r5 = ConstraintList()
        self.model.r6 = ConstraintList()
        self.model.r7 = ConstraintList()

        for s in range(S):
            # R1
            self.model.r1.add(
                sum(self.model.t[l, s] for l in ins.L) == ins.h[s] * (1 - self.model.n[s]))

            # R4
            self.model.r4.add(self.model.y[s] == sum(
                self.model.x[l, s] for l in ins.L))

            # R5
            self.model.r5.add(1 - self.model.y[s] <= self.model.n[s])

            for l in ins.L:
                # R2
                self.model.r2.add(
                    ins.tmin * self.model.x[l, s] <= self.model.t[l, s])
                self.model.r2.add(
                    BIG_M * self.model.x[l, s] >= self.model.t[l, s])

                # R3
                if ins.r[l, ins.ids[s]] == 0:
                    self.model.r3.add(self.model.x[l, s] == 0)

                # R5
                self.model.r5.add(self.model.n[s] <= 1 - self.model.x[l, s])
    
        # R6
        for l in ins.L:
            for p in range(1, ins.P + 1):
                self.model.r6.add(self.model.u[l, p] == ins.d[l, p] - sum(self.model.t[l, s] for s in ins.active[0, p]))
        if ins.mode == 'saa':
            for l in ins.L:
                for p in range(1, ins.T + 1):
                    for e in range(1, ins.E + 1):
                        self.model.r6.add(self.model.z[l, e, p] == self.model.u[l, p] - sum(self.model.t[l, s] for s in ins.active[e, p]))

        # R7
        for s in range(S):
            self.model.r7.add(self.model.R[s] == sum(self.model.t[l, s] * ins.H[s] * ins.r[l, ins.ids[s]] for l in ins.L))

        # Guardamos la instancia para después
        self.instance = ins

    def run(self, solver='glpk', time_limit=300):
        '''
        Función para correr el modelo después de creado
        -----------------------------------------------------------------------------------------------------------------
        INPUT:
            solver: por defecto se selecciona glpk como solver, pero puede ser cualquiera dentro de las opciones de Pyomo
            time_limit: límite de tiempo para resolver el modelo
        OUTPUT:
            assignment: lista de listas tal que assignment[i] es una lista de tuplas (l, t) asociadas al servicio
                        i (l corresponde al abogado y t al tiempo asignado)
            time_left: diccionario de key (l, p) y value tiempo en horas
                       disponibles del abogado l en el periodo p
            sr: lista tal que sr[i] es el rating de asignación del
                servicio i
        '''
        
        #### ACÁ VA LA CONFIGURACIÓN DEL SOLVER ####
        opt = SolverFactory(solver)
        opt.options['timelimit'] = time_limit
        ###########################################

        opt.solve(self.model)

        # Se construyen estructuras a retornar
        ins = self.instance
        # Asignaciones
        assignment = []
        # Tiempo que queda de abogados después de asignación
        time_left = ins.d
        # Rating de servicios
        sr = []

        for l in ins.L:
            for p in range(1, ins.P + 1):
                time_left[l, p] = value(self.model.u[l, p])

        for s in range(ins.S_0):
            a = []
            sr.append(value(self.model.R[s]))
            for l in ins.L:
                if value(self.model.x[l, s]):
                    a.append((l, value(self.model.t[l, s])))
            assignment.append(a)

        return assignment, time_left, sr
