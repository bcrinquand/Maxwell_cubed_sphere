from sympy import *
import numpy as N
from sympy import MatrixSymbol, Matrix

delta, gam  = symbols('delta gam')
# Ja32, Ja12, Ja0, Jb0, Jb12, Jb32  = symbols('Ja32 Ja12 Ja0 Jb0 Jb12 Jb32') 
# Ja32f, Ja12f, Ja0f, Jb0f, Jb12f, Jb32f  = symbols('Ja32f Ja12f Ja0f Jb0f Jb12f Jb32f') 

# Ta32, Ta12, Ta0, Tb0, Tb12, Tb32  = symbols('Ta32 Ta12 Ta0 Tb0 Tb12 Tb32') 
# Ta32f, Ta12f, Ta0f, Tb0f, Tb12f, Tb32f  = symbols('Ta32f Ta12f Ta0f Tb0f Tb12f Tb32f')

x41, x42, x43, x44, x45, x46, x51, x52, x53, x54, x55, x56, x64, x65, x66, x67  = symbols('x41 x42 x43 x44 x45 x46 x51 x52 x53 x54 x55 x56 x64 x65 x66 x67') 

Filt = Matrix([[x66,x65,x64,0.0,0.0,0.0], \
			   [x56,x55,x54,x53,0.0,0.0], \
			   [0.0,x45,x44,x43,0.0,0.0], \
			   [0.0,0.0,x43,x44,x45,0.0], \
			   [0.0,0.0,x53,x54,x55,x56], \
			   [0.0,0.0,0.0,x64,x65,x66], \
		])

# Filt = Matrix([[x66,x65,x64,0.0,0.0,0.0,0.0], \
# 			   [x56,x55,x54,x53,x52,x51,0.0], \
# 			   [x46,x45,x44,x43,x42,x41,0.0], \
# 			   [x41,x42,x43,x44,x45,x46,0.0], \
# 			   [x51,x52,x53,x54,x55,x56,0.0], \
# 			   [0.0,0.0,0.0,x64,x65,x66,x67], \
# 			   [0.0,0.0,0.0,0.0,0.0,0.25,0.5], \
# 		])

A = Matrix([[2*delta-2, 2-3*delta], \
            [2*gam-2  , 2-3*gam],  \
           ])

Ainv = A**(-1)

Jb0f  = simplify(Ainv[0, 0] * (0.25 - delta * 0.5) + Ainv[0, 1] * (- 0.25 - gam * 0.5))
Jb12f = simplify(Ainv[1, 0] * (0.25 - delta * 0.5) + Ainv[1, 1] * (- 0.25 - gam * 0.5))

# Jb0f  = expand(((3 * gam - 2) * (0.25 - 0.5 * delta) + (2 - 3 * delta) * (0.25 - 0.5 * gam)) / 2.0 / (delta - gam))
# Jb12f = expand((2 * (gam - 1) * (0.25 - 0.5 * delta) + 2 * (1 - delta) * (0.25 - 0.5 * gam)) / 2.0 / (delta - gam))
Jb32f = 0.5
Ja32f = 0.0
# Ja12f = expand((1 - gam) / 4 / (gam - delta))
# Ja0f  = expand((2 * gam - 3) / 8 / (gam - delta))
Ja0f  = simplify(Ainv[0, 0] * (- 0.25) + Ainv[0, 1] * (- 0.25))
Ja12f = simplify(Ainv[1, 0] * (- 0.25) + Ainv[1, 1] * (- 0.25))
Jb52f = 0.25

dqa0f = simplify(+ (2*(delta-1)*Ja0f + (2-3*delta)*Ja12f + delta*Ja32f))
dqb0f = simplify(- (2*(delta-1)*Jb0f + (2-3*delta)*Jb12f + delta*Jb32f))
dqb12f = simplify(- (2*(gam-1)*Jb0f + (2-3*gam)*Jb12f + gam*Jb32f))

Ja0  = 0.0
Ja12 = 0.0
Ja32 = 0.0
Jb0  = expand((delta*(gam-delta)+2*gam-2-3*delta*gam+3*delta) / 2 / (1-delta) / (gam-delta))
Jb12 = expand((gam - 1.0) / (gam - delta))
Jb32 = 1.0
Jb52 = 0.0

dqa0 = simplify(+ (2*(delta-1)*Ja0 + (2-3*delta)*Ja12 + delta*Ja32))
dqb0 = simplify(- (2*(delta-1)*Jb0 + (2-3*delta)*Jb12 + delta*Jb32))
dqb12 = simplify(- (2*(gam-1)*Jb0 + (2-3*gam)*Jb12 + gam*Jb32))

J = Matrix([Ja32, Ja12, Ja0, Jb0, Jb12, Jb32])

Tb0f  = expand(((3 * gam - 2) * (0.25 - 0.25 * delta) + (2 - 3 * delta) * (- 0.25 - 0.25 * gam)) / 2.0 / (delta - gam))
Tb12f = expand((2 * (gam - 1) * (0.25 - 0.25 * delta) + 2 * (1 - delta) * (- 0.25 - 0.25 * gam)) / 2.0 / (delta - gam))
Tb32f = 0.25
Ta32f = 0.0
Ta12f = expand((2 * (gam - 1) * (- 0.25) + 2 * (1 - delta) * (- 0.25)) / 2.0 / (delta - gam))
Ta0f  = expand(((3 * gam - 2) * (- 0.25) + (2 - 3 * delta) * (- 0.25)) / 2.0 / (delta - gam))
Tb52f = 0.0

dta0f = simplify(+ (2*(delta-1)*Ta0f + (2-3*delta)*Ta12f + delta*Ta32f))
dtb0f = simplify(- (2*(delta-1)*Tb0f + (2-3*delta)*Tb12f + delta*Tb32f))
dtb12f = simplify(- (2*(gam-1)*Tb0f + (2-3*gam)*Tb12f + gam*Tb32f))

Ta0  = expand((3.0 * gam - 2.0) / (2.0 * gam - 2.0 * delta))
Ta12 = expand((gam - 1.0) / (gam - delta))
Ta32 = 0.0
Tb0  = expand((3.0 * delta + 3.0 * gam - 4.0) / 2.0 / (delta - gam))
Tb12 = expand((delta + gam - 2.0) / (delta - gam))
Tb32 = 0.0
Tb52 = 0.0

dta0 = simplify(+ (2*(delta-1)*Ta0 + (2-3*delta)*Ta12 + delta*Ta32))
dtb0 = simplify(- (2*(delta-1)*Tb0 + (2-3*delta)*Tb12 + delta*Tb32))
dtb12 = simplify(- (2*(gam-1)*Tb0 + (2-3*gam)*Tb12 + gam*Tb32))

T = Matrix([Ta32, Ta12, Ta0, Tb0, Tb12, Tb32])

eqJ0 = Eq(Ja32f, simplify((Filt[0, :] * J)[0,0]))
eqJ1 = Eq(Ja12f, simplify((Filt[1, :] * J)[0,0]))
eqJ2 = Eq(Ja0f,  simplify((Filt[2, :] * J)[0,0]))
eqJ3 = Eq(Jb0f,  simplify((Filt[3, :] * J)[0,0]))
eqJ4 = Eq(Jb12f, simplify((Filt[4, :] * J)[0,0]))
eqJ5 = Eq(Jb32f, simplify((Filt[5, :] * J)[0,0]))
# eqJ6 = Eq(Jb52f, simplify((Filt[6, :] * J)[0,0]))

eqT0 = Eq(Ta32f, simplify((Filt[0, :] * T)[0,0]))
eqT1 = Eq(Ta12f, simplify((Filt[1, :] * T)[0,0]))
eqT2 = Eq(Ta0f,  simplify((Filt[2, :] * T)[0,0]))
eqT3 = Eq(Tb0f,  simplify((Filt[3, :] * T)[0,0]))
eqT4 = Eq(Tb12f, simplify((Filt[4, :] * T)[0,0]))
eqT5 = Eq(Tb32f, simplify((Filt[5, :] * T)[0,0]))
# eqT6 = Eq(Tb52f, simplify((Filt[6, :] * T)[0,0]))

result = solve([eqJ0, eqJ1, eqJ2, eqJ3, eqJ4, eqJ5, eqT0, eqT1, eqJ2, eqT3, eqT4, eqT5], \
                (x44, x45, x53, x54, x55, x56, x64, x65, x66, x43))

# fil = Filt.subs([(x42, simplify(result[x42])), (x44, simplify(result[x44])), (x45, simplify(result[x45])), \
#                  (x53, simplify(result[x53])), (x54, simplify(result[x54])), (x55, simplify(result[x55])), \
#                  (x56, simplify(result[x56])), (x64, simplify(result[x64])), (x65, simplify(result[x65])), \
#                  (x66, simplify(result[x66]))])

# jxijf = simplify(jxij.subs([(alpha, result[alpha]), (beta, result[beta])]))
# jxipjf = simplify(jxipj.subs([(alpha, result[alpha]), (beta, result[beta])]))
# jxijpf = simplify(jxijp.subs([(gamma, result[gamma]), (delta, result[delta])]))
# jxipjpf = simplify(jxipjp.subs([(gamma, result[gamma]), (delta, result[delta])]))
