import pickle 

with open(f'results/sa.pickle', 'rb') as file:
    sa = pickle.load(file)

with open(f'results/time_lawyers.pickle', 'rb') as file:
    tl = pickle.load(file)

with open(f'results/assignments.pickle', 'rb') as file:
    a = pickle.load(file)

print('SAA')
print(a[4][0.8]['saa'])
print('\n')

print('GREEDY')
print(a[4][0.8]['greedy'])

# for w in a[4][0.8]['saa']:
#     print(f'SEMANA {w}')
#     print(sa[4][0.8]['saa'][w])
#     print(a[4][0.8]['saa'][w])
#     print('\n')
