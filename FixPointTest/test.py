from math import sqrt  
N = 100 
x = [p for p in range(2 ** 20-100, 2 ** 20) if 0 not in [ p% d for d in range(2, int(sqrt(p))+1)] ]

print(x)