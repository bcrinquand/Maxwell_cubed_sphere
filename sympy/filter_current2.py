from sympy import *
import numpy as N
import matplotlib.pyplot as P 
P.ion()

delta, gam  = symbols('delta gam', real=True)

c00, c10, c20, c30, c40, c01, c11, c21, c31, c41, c02, c12, c22, c32, x00, x01, x02, x52, x51, x50, x10, x11, x12, x42, x41, x40, x20, x21, x22, x30, x32, xe, xf, xg, ce, cf, cg, ch  = \
    symbols('c00 c10 c20 c30 c40 c01 c11 c21 c31 c41 c02 c12 c22 c32 x00 x01 x02 x52 x51 x50 x10 x11 x12 x42 x41 x40 x20 x21 x22 x30 x32 xe xf xg ce cf cg ch') 

c12 = 0.0

c11p = 1 - 0.25 - c10
c00p = 0.5 #1 - c10 - c40
# c50 = c00p
c50 = 0.0
c40 = 0.0

Filt_q = Matrix([[0.5,0.5,0.0,0.00,0.00,c40,c50], \
			     [0.25,0.5,0.25,0.00,0.00,c12,c40], \
			     [0.0,0.25,0.50,0.25,0.00,0.00,0.0], \
			     [0.00,0.00,0.25,0.50,0.25,0.00,0.00], \
			     [0.00,0.00,0.00,0.25,0.50,0.25,0.0], \
			     [c40,c12,0.00,0.00,0.25,0.5,0.25], \
			     [c50,c40,0.0,0.00,0.0,0.5,0.5], \
		])

# c11p = 1 - 0.25 - c10
# c00p = 1 - c10

# Filt_q = Matrix([[c00p,c10,0.0,0.00,0.00,0.0,c00p], \
# 			     [c10,c11p,0.25,0.00,0.00,0.0,0.0], \
# 			     [0.0,0.25,0.50,0.25,0.00,0.00,0.0], \
# 			     [0.00,0.00,0.25,0.50,0.25,0.00,0.00], \
# 			     [0.00,0.00,0.00,0.25,0.50,0.25,0.0], \
# 			     [0.0,0.0,0.00,0.00,0.25,c11p,c10], \
# 			     [c00p,0.0,0.0,0.00,0.0,c10,c00p], \
# 		])

# c00 = 0.75

# Filt_q = Matrix([[c00,c10,0.00,0.00,0.00,0.0,0.0], \
# 			     [c10,0.50,0.25,0.00,0.00,0.00,0.0], \
# 			     [0.00,0.25,0.50,0.25,0.00,0.00,0.00], \
# 			     [0.00,0.00,0.25,0.50,0.25,0.00,0.00], \
# 			     [0.00,0.00,0.00,0.25,0.50,0.25,0.00], \
# 			     [0.0,0.00,0.00,0.00,0.25,0.50,c10], \
# 			     [0.0,0.0,0.00,0.00,0.0,c10,c00], \
# 		])

# Filt_j = Matrix([[x00,x01,x02 ,xf  ,0.00,0.0 ,0.0,x50], \
# 			     [x10,x11,x12 ,xe  ,0.00,0.0 ,0.0,x40], \
# 			     [0.0,0.25,0.5,0.25,0.00,0.00,0.0,0.0], \
# 			     [0.0,0.0,0.25,0.50,0.25,0.00,0.0,0.0], \
# 			     [0.0,0.0,0.00,0.25,0.50,0.25,0.0,0.0], \
# 			     [0.0,0.0,0.00,0.00,0.25,0.5,0.25,0.0], \
# 			     [x40,0.0,0.0 ,0.00,xf, x12 ,x11,x10], \
# 			     [x50,0.0,0.0 ,0.00,xe, x02 ,x01,x00], \
# 		])

Filt_j = Matrix([[x00,x01,x02 ,xf,0.00,x52,x51,x50], \
			     [x10,x11,x12 ,xe,0.00,x42,x41,x40], \
			     [x20,x21,x22,0.25,0.00,0.00,0.0,0.0], \
			     [0.0,0.0,0.25,0.50,0.25,0.00,0.0,0.0], \
			     [0.0,0.0,0.00,0.25,0.50,0.25,0.0,0.0], \
			     [0.0,0.0,0.00,0.00,0.25,x22 ,x21,x20], \
			     [x40,x41,x42 ,0.00,xe, x12 ,x11,x10], \
			     [x50,x51,x52 ,0.00,xf, x02 ,x01,x00], \
		])

# delta=0.5
# gam=0.75

a1=2.0*(gam-1.0)
a2 = 2.0-3.0*gam

b1=2.0*(delta-1.0)
b2 = 2.0-3.0*delta

Dp = Matrix([[b1, b2, delta,0.0,0.0,0.0,0.0,0.0], \
			 [a1, a2, gam  ,0.0,0.0,0.0,0.0,0.0], \
			 [0.0,0.0,-1.0 ,1.0,0.0,0.0,0.0,0.0], \
			 [0.0,0.0,0.0,-1.0,1.0,0.0,0.0,0.0], \
			 [0.0,0.0,0.0,0.0,-1.0,1.0,0.0,0.0], \
			 [0.0,0.0,0.0,0.0,0.0,-gam,-a2,-a1], \
			 [0.0,0.0,0.0,0.0,0.0,-delta,-b2,-b1]
		])

Dpy = Matrix([[b1, b2, delta,0.0,0.0,0.0,0.0,0.0], \
			 [a1, a2, gam  ,0.0,0.0,0.0,0.0,0.0], \
			 [0.0,0.0,-1.0 ,1.0,0.0,0.0,0.0,0.0], \
			 [0.0,0.0,0.0,-1.0,1.0,0.0,0.0,0.0], \
			 [0.0,0.0,0.0,0.0,-1.0,1.0,0.0,0.0], \
			 [0.0,0.0,0.0,0.0,0.0,-gam,-a2,-a1], \
			 [0.0,0.0,0.0,0.0,0.0,-delta,-b2,-b1]
		])

prod=simplify(Filt_q*Dp-Dp*Filt_j)
prod1=simplify(Filt_q*Dp)
prod2=simplify(Dp*Filt_j)

# for i in range(3):
#     for j in range(8):
        
#         name 

eq00 = Eq(0.0, prod[0,0])
eq01 = Eq(0.0, prod[0,1])
eq02 = Eq(0.0, prod[0,2])
eq03 = Eq(0.0, prod[0,3])
eq04 = Eq(0.0, prod[0,4])
eq05 = Eq(0.0, prod[0,5])
eq06 = Eq(0.0, prod[0,6])
eq07 = Eq(0.0, prod[0,7])

eq10 = Eq(0.0, prod[1,0])
eq11 = Eq(0.0, prod[1,1])
eq12 = Eq(0.0, prod[1,2])
eq13 = Eq(0.0, prod[1,3])
eq14 = Eq(0.0, prod[1,4])
eq15 = Eq(0.0, prod[1,5])
eq16 = Eq(0.0, prod[1,6])
eq17 = Eq(0.0, prod[1,7])

eq20 = Eq(0.0, prod[2,0])
eq21 = Eq(0.0, prod[2,1])
eq22 = Eq(0.0, prod[2,2])
eq23 = Eq(0.0, prod[2,3])
eq24 = Eq(0.0, prod[2,4])
eq25 = Eq(0.0, prod[2,5])
eq26 = Eq(0.0, prod[2,6])
eq27 = Eq(0.0, prod[2,7])

result = solve([eq00, eq01, eq02, eq03, eq04, eq05, eq06, eq07, eq10, eq11, eq12, eq13, eq14, eq15, eq16, eq17, eq20, eq21, eq22, eq23, eq24, eq25, eq26, eq27], \
                (x00, x01, x02, x52, x51, x50, x10, x11, x12, x41, x40, x20, x21, x22, x30, x32, x42, xe, xf, c00, c10, c20, c11, c12))

prodf = simplify(prod.subs([(x00, result[x00]), (x10, result[x10]), (x20, result[x20]), (x40, result[x40]), (x50, result[x50]), \
                     (x01, result[x01]), (x11, result[x11]), (x21, result[x21]), (x52, result[x52]), (x51, result[x51]),  \
                     (x02, result[x02]), (x12, result[x12]), (x22, result[x22]), (x42, result[x42]), (x41, result[x41]), (xe, result[xe]), (xf, result[xf])]))

c10f = 0.25
c12f = 0.0
gamf = 1.0 - 0.5 * delta #0.75

x10f = simplify(result[x10].subs([(gam, gamf), (c10, c10f), (c12, c12f)]))
x20f = simplify(result[x20].subs([(gam, gamf), (c10, c10f), (c12, c12f)]))
x40f = simplify(result[x40].subs([(gam, gamf), (c10, c10f), (c12, c12f)]))
x50f = simplify(result[x50].subs([(gam, gamf), (c10, c10f), (c12, c12f)]))
x01f = simplify(result[x01].subs([(gam, gamf), (c10, c10f), (c12, c12f)]))
x11f = simplify(result[x11].subs([(gam, gamf), (c10, c10f), (c12, c12f)]))
x21f = simplify(result[x21].subs([(gam, gamf), (c10, c10f), (c12, c12f)]))
x00f = simplify(result[x00].subs([(gam, gamf), (c10, c10f), (c12, c12f)]))
x02f = simplify(result[x02].subs([(gam, gamf), (c10, c10f), (c12, c12f)]))
x12f = simplify(result[x12].subs([(gam, gamf), (c10, c10f), (c12, c12f)]))
x22f = simplify(result[x22].subs([(gam, gamf), (c10, c10f), (c12, c12f)]))
x41f = simplify(result[x41].subs([(gam, gamf), (c10, c10f), (c12, c12f)]))
x51f = simplify(result[x51].subs([(gam, gamf), (c10, c10f), (c12, c12f)]))
x42f = simplify(result[x42].subs([(gam, gamf), (c10, c10f), (c12, c12f)]))
x52f = simplify(result[x52].subs([(gam, gamf), (c10, c10f), (c12, c12f)]))
xef  = simplify(result[xe].subs([ (gam, gamf), (c10, c10f), (c12, c12f)]))
xff  = simplify(result[xf].subs([ (gam, gamf), (c10, c10f), (c12, c12f)]))

plot(x00f, x10f, x20f, xef, x01f, x11f, x21f, xff, x02f, x12f, x22f, (delta, 0.0, 0.6), legend = True)

plot(x40f, x50f, (delta, 0.0, 0.6))
plot(x41f, x51f, (delta, 0.0, 0.6))
plot(x42f, x52f, (delta, 0.0, 0.6))

deltaf = 0.5
Filt_jf = Filt_j.subs([(x00, x00f), (x10, x10f), (x20, x20f), (x40, x40f), (x50, x50f), \
                       (x01, x01f), (x11, x11f), (x21, x21f), (xe, xef), (xf, xff), \
                       (x02, x02f), (x12, x12f), (x22, x22f), (x40, x40f), (x41, x41f), (x42, x42f), \
                        (x50, x50f), (x51, x51f), (x52, x52f), (delta, deltaf)
    ])

Filt_qf = Filt_q.subs([(c10, c10f), (c12, c12f)])
