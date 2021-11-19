from gurobipy import *

m = read("asignacion_abogados.lp")
m.optimize()

for v in m.getVars():
    if "x" in v.VarName:
        s = v.VarName
        l = s.split("_")
        if int(l[-1]) < 10:
            if v.x > 0:
                print(v.VarName, "=", v.x)

