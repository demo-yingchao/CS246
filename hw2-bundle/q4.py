import numpy as np


f = open('q4/data/user-shows.txt', 'r')
R = []
while True:
  l = f.readline()
  if not l:
    break
  l = [int(r) for r in l[:-1].split(' ')]
  R.append(l)
R = np.array(R)
print(R.shape)
P = np.diag(np.sum(R, 1))
Q = np.diag(np.sum(R, 0))

P_root = np.diag(np.sqrt(1.0/np.sum(R, 1)))
Q_root = np.diag(np.sqrt(1.0/np.sum(Q, 1)))

uu = P_root@R@R.T@P_root@R
mm = R@Q_root@R.T@R@Q_root
_R = R[:, 100:]
deg = np.sqrt(np.sum(_R, 1))
deg += (deg == 0)*1e-6
_P_root = np.diag(1.0/deg)
_uu =  _P_root@_R@_R.T@R[:, 0:100]

suu = np.argsort(uu[499][0:100], kind='mergesort')
smm = np.argsort(mm[499], kind='mergesort')
_suu = np.argsort(_uu[499], kind='mergesort')
print('normal user-user')
print(suu[-1:-6:-1])
print('special user-user')
print(_suu[-1:-6:-1])
print('movie-movie')
print(smm[-1:-6:-1])

print('user-user max all')
print(np.max(uu))
print('user-user max Alex')
print(np.max(uu[499]))
print('movie-movie max all')
print(np.max(mm))
print('movie-movie max Alex')
print(np.max(mm[499]))
