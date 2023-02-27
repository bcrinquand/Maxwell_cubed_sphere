from sympy import *

hrru, hr1u, hr2u, h11u, h12u, h22u, betard, alpha = symbols('hrru hr1u hr2u h11u h12u h22u betard alpha') 

inv_metric = Matrix([[hrru, hr1u, hr2u], \
				 	 [hr1u, h11u, h12u], \
				 	 [hr2u, h12u, h22u]])

metric = inv_metric.inv()
sqrt_det_h = det(metric)

hrrd = metric[0, 0]
hr1d = metric[1, 0]
hr2d = metric[2, 0]
h11d = metric[1, 1]
h12d = metric[1, 2]
h22d = metric[2, 2]

#M = Matrix([[0,    -betard,    0,  hr2d,      h12d,  h22d], \
#			[0,          0,    0,     0,         0,     0], \
#			[0,          0,    0, -hrrd,     -hr1d, -hr2d], \
#			[-hr2d,  -h12d,-h22d,     0,   -betard,    0], \
#			[0,          0,    0,     0,         0,    0], \
#			[hrrd,    hr1d,  hr2d,    0,         0,    0]])

M = Matrix([[0,    -betard,    0,  hr2u,      h12u,  h22u], \
			[0,          0,    0,     0,         0,     0], \
			[0,          0,    0, -hrru,     -hr1u, -hr2u], \
			[-hr2u,  -h12u,-h22u,     0,   -betard,    0], \
			[0,          0,    0,     0,         0,    0], \
			[hrru,    hr1u,  hr2u,    0,         0,    0]])


P, D = M.diagonalize()

Dr, D1, D2, Br, B1, B2 = symbols('Dr D1 D2 Br B1 B2') 
vec = Matrix([Dr, D1, D2, Br, B1, B2])

#dp = det(P)

#I = P.inv()

#print(I*vec)

