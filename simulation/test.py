import pickle 

with open(f'results/sr.pickle', 'rb') as file:
    sr = pickle.load(file)

with open(f'results/time_lawyers.pickle', 'rb') as file:
    tl = pickle.load(file)

with open(f'results/assignments.pickle', 'rb') as file:
    a = pickle.load(file)

print('SAA')
print(a[3][0.8]['saa'])
print('\n')

print('GREEDY')
print(a[3][0.8]['greedy'])
