import pickle 

with open(f'results/sa.pickle', 'rb') as file:
    sa = pickle.load(file)

with open(f'results/time_lawyers.pickle', 'rb') as file:
    tl = pickle.load(file)

with open(f'results/assignments.pickle', 'rb') as file:
    a = pickle.load(file)

# print('SAA')
# print([1.25][0.8]['saa'])
# print('\n')

# print('GREEDY')
# print(tl[1.25][0.8]['greedy'])

for w in a[1.25][0.8]['saa']:
    print(f'SEMANA {w}')
    print(sa[1.25][0.8]['saa'][w])
    print(a[1.25][0.8]['saa'][w])
    print('\n')
