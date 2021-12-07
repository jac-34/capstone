import pickle

with open(f'resultados/pond10/rp.pickle', 'rb') as file:
    rp = pickle.load(file)
with open(f'resultados/pond10/rp_greedy.pickle', 'rb') as file:
    rp_greedy = pickle.load(file)

with open(f'resultados/pond10/nasp.pickle', 'rb') as file:
    nasp = pickle.load(file)
with open(f'resultados/pond10/nasp_greedy.pickle', 'rb') as file:
    nasp_greedy = pickle.load(file)

with open(f'resultados/pond10/fssa.pickle', 'rb') as file:
    fssa = pickle.load(file)
with open(f'resultados/pond10/fssa_greedy.pickle', 'rb') as file:
    fssa_greedy = pickle.load(file)



print(fssa)

print('\n')

print(fssa_greedy)

