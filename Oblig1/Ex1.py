from dolfin import *

set_log_active(False)
import numpy as np


def Hp(k, l, p):
	s = 0
	for i in range(p+1):
		for j in range(p+1-i):
			s += 0.5 * (k*np.pi)**i * (l*np.pi)**j
	return s

def bc1(x,on_boundary):
	return (near(x[0], 0) or near(x[0], 1)) and on_boundary


k1 = [1, 10, 100]
l1 = [1, 10, 100]
h1 = [8, 16, 32, 64]
Pe = [1,2]

H1 = []
L2 = []

for p in Pe:
	bd = open('table%s.txt' %p,'w') 
	bd.write('\\begin{tabular}{ |c|c|c|c|c| }\n')
	bd.write('\hline%cline{1-1} \cline{4-5}\n')
	bd.write('h & l & k &$L^2$&$H^1$ \\\\\n')
	bd.write('\hline \n')

	H = []
	L = []
	meshsize = []
	for h in h1:
		bd.write('\multirow{9}{*}{\\textbf{%s}}' %h) 
		for l in l1:
			bd.write(' & \multirow{3}{*}{%s} ' %l)
			for k in k1:

				mesh = UnitSquareMesh(h, h)
				meshsize.append(mesh.hmin())

				V = FunctionSpace(mesh, 'CG', p)
				V2 = FunctionSpace(mesh, 'CG', p+2)

				bc = DirichletBC(V, Constant(0), bc1)
				u = TrialFunction(V)
				v = TestFunction(V)

				f = Expression( 'pi*pi * sin(k*pi*x[0])*cos(l*pi*x[1])* (k*k + l*l)', k = k, l = l)
				
				F = inner(grad(u), grad(v))*dx - f*v*dx

				u_ = Function(V)

				solve(lhs(F) == rhs(F), u_, bc)

				u_ex = Expression('sin(k*pi*x[0])*cos(l*pi*x[1])',k = k, l = l)
				u_e = interpolate(u_ex, V2)

				ud= abs(u_ - u_e)
				
				#L2_norm = errornorm(u_, u_e)
				L2_norm = sqrt(assemble(ud**2*dx))

				#H1_norm= errornorm(u_, u_e, 'H1')
				H1_norm = sqrt(assemble(ud*ud*dx+ inner(grad(ud), grad(ud))*dx))
				
				H.append(H1_norm)
				L.append(L2_norm)


				if k ==1:
					bd.write('& %s   & %0.5f  & %0.5f \\\\ \cline{3-5}\n' %(k, L2_norm, H1_norm))
				elif k ==100:
					bd.write('&& %s   & %0.5f  & %0.5f \\\\ \cline{2-5}\n' %(k, L2_norm, H1_norm))
				else:
					bd.write('&& %s   & %0.5f  & %0.5f \\\\ \cline{3-5}\n' %(k, L2_norm, H1_norm))
		
		bd.write('\\hline \\hline \n')
	H1.append(H)
	L2.append(L)
	H1.append(meshsize)


	bd.write('\\end{tabular}')
	bd.close()

for j in [H1, L2]:
	
	for i in range(2):
		A = np.zeros((2,2))
		b = np.zeros(2)
		A[0][0] = len(H1[i*2])
		A[0][1] = sum(np.log(H1[2*i+1])) 
		
		A[1][0] = sum(np.log(H1[2*i+1]))
		A[1][1] = sum(np.log(H1[2*i+1])**2)
		if j == L2:
			b[1] = sum(np.log(H1[2*i+1])*np.log(j[i]))
			b[0] = sum(np.log(j[i]))
		else:
			b[1] = sum(np.log(H1[2*i+1])*np.log(j[2*i]))
			b[0] = sum(np.log(j[2*i]))
		logC, alpha = np.linalg.solve(A,b)
		print 'alpha/beta:', alpha, 'C:', np.exp(logC)

#print A, hi