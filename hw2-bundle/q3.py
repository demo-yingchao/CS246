import numpy as np
import matplotlib.pyplot as plt

f = open('q3/data/ratings.train.txt', 'r')
user_max = 0
movie_max = 0
user_min = 1e7
movie_min = 1e7
count = 0
while 1:
  l = f.readline()
  if not l:
    break
  l = [int(x) for x in l[:-1].split('\t')]
  if l[0] > user_max:
    user_max = l[0]
  if l[0] < user_min:
    user_min = l[0]
  if l[1] > movie_max:
    movie_max = l[1]
  if l[1] < movie_min:
    movie_min = l[1]
  count += 1

print('user (%d, %d)' %(user_min, user_max))
print('movie (%d, %d)' %(movie_min, movie_max))
print('line count %d' % count)

iterations = 40
k = 30
lam = 0.1
lr = 0.01

q_len = user_max
p_len = movie_max

q = np.random.uniform(0, np.sqrt(5/k), (q_len, k))
p = np.random.uniform(0, np.sqrt(5/k), (p_len, k))

E = []
f.seek(0)

for _ in range(iterations):
  f.seek(0)
  while True:
    l = f.readline()
    if not l:
      break
    l = [int(x) for x in l[:-1].split('\t')]
    i = l[0] - 1
    u = l[1] - 1
    e = l[2] - q[i]@p[u]
    dq = e*p[u]
    dp = e*q[i]
    q *= (1 - lr*lam/count)
    p *= (1 - lr*lam/count)
    q[i] += lr*dq
    p[u] += lr*dp

  f.seek(0)
  err = 0
  while True:
    l = f.readline()
    if not l:
      break
    l = [int(x) for x in l[:-1].split('\t')]
    i = l[0] - 1
    u = l[1] - 1
    err += (l[2] - q[i]@p[u])**2

  err += lam * (np.sum(p**2)+np.sum(q**2))
  E.append(float(err))

print(E)
plt.figure('E')
plt.plot(E, '-.^')
plt.show()
  
