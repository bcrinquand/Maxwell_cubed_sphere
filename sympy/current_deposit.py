from sympy import *

x1,x2,y1,y2,dx,dy = symbols('x1 x2 y1 y2 dx dy') 

Fx = x2 - x1
Fy = y2 - y1
Wx = 0.5 * (x2 + x1)
Wy = 0.5 * (y2 + y1)

Jxij  = Fx * (1.0 - Wy)
Jxijp = Fx * Wy
Jyij  = Fy * (1.0 - Wx)
Jyipj = Fy * Wx


delta1 = simplify(Jxij + Jyij)
delta2 = simplify(- Jxij + Jyipj)
delta3 = simplify(Jxijp - Jyij)
delta4 = simplify(- Jxijp - Jyipj)

#print(delta1, '=?', -expand((1.0 - x2)*(1.0 - y2) - (1.0 - x1)*(1.0 - y1)))
#print(delta2, '=?', -expand(x2*(1.0 - y2) - x1*(1.0 - y1)))
#print(delta3, '=?', -expand((1.0 - x2)*y2 - (1.0 - x1)*y1))
#print(delta4, '=?', -(x2*y2 - x1*y1))

alpha, beta, gamma, delta = symbols('alpha beta gamma delta')

#jxij  = (x2 - x1) * (alpha * y1 + beta * y2)
#jxipj = (x2 - x1) * (gamma * y1 + delta * y2)
#jxijp  = (x2 - x1) * (1.0 - alpha * y1 - beta * y2)
#jxipjp = (x2 - x1) * (1.0 - gamma * y1 - delta * y2)
#jyij  = (y2 - y1) * (1.0 - 0.5 * (x2 + x1))
#jyijp = (y2 - y1) * 0.5 * (x2 + x1)

#jxij  = 0.0 
#jxipj = (x2 - x1) * (alpha * y1 + beta * y2)
#jxijp  = 0.0 
#jxipjp = (x2 - x1) * (1.0 - gamma * y1 - delta * y2)
#jyij  = (y2 - y1) * (1.0 - 0.5 * (x2 + x1))
#jyijp = (y2 - y1) * 0.5 * (x2 + x1)


jxij  = (x2 - x1) * (1.0 - alpha * (y1 + y2))
jxipj = (x2 - x1) * (1.0 - beta * (y1 + y2))
jxijp  = (x2 - x1) * (gamma * (y1 + y2))
jxipjp = (x2 - x1) * (delta * (y1 + y2))

jyij  = (y2 - y1) * (1.0 - 0.5 * (x2 + x1))
jyijp = (y2 - y1) * 0.5 * (x2 + x1)

eq1 = Eq(-0.5 * jxipj + jxij - jyij, -expand((1.0 - x2)*(1.0 - y2) - (1.0 - x1)*(1.0 - y1)))
eq2 = Eq(-0.5 * jxipjp + jxijp + jyij, -expand((1.0 - x2)*y2 - (1.0 - x1)*y1))
eq3 = Eq( 0.25 * jxipj  + 0.5 * jxij  - jyijp, -expand(x2*(1.0 - y2) - x1*(1.0 - y1)))
eq4 = Eq( 0.25 * jxipjp + 0.5 * jxijp + jyijp, -(x2*y2 - x1*y1))

result = solve([eq1, eq2, eq3, eq4], (alpha, beta, gamma, delta))


jxijf = simplify(jxij.subs([(alpha, result[alpha]), (beta, result[beta])]))
jxipjf = simplify(jxipj.subs([(alpha, result[alpha]), (beta, result[beta])]))
jxijpf = simplify(jxijp.subs([(gamma, result[gamma]), (delta, result[delta])]))
jxipjpf = simplify(jxipjp.subs([(gamma, result[gamma]), (delta, result[delta])]))

print(jxijf)
print(jxipjf)
print(jxijpf)
print(jxipjpf)

a=0.5 * (x2 - x1) * (1.0 - 0.5 * (y1 + y2))
b=3.0 * (x2 - x1) * (1.0 - 0.5 * (y1 + y2))
c=0.5 * (x2 - x1) * (y1 + y2) / 2.0
d=3.0 * (x2 - x1) * (y1 + y2) / 2.0

#jxijf = simplify(-a+0.5*b+jyij + (1.0 - x2)*(1.0 - y2) - (1.0 - x1)*(1.0 - y1))
#jxipjf = simplify(-0.5*a-0.25*b+jyijp + x2*(1.0 - y2) - x1*(1.0 - y1))
#jxijpf = simplify(-c+0.5*d-jyij + (1.0 - x2)*y2 - (1.0 - x1)*y1)
#jxipjpf = simplify(-0.5*c-0.25*d-jyijp + x2*y2 - x1*y1)

#print(jxijf)
#print(jxipjf)
#print(jxijpf)
#print(jxipjpf)


