from dolfin import *
set_log_active(False)
import numpy as np

def alpha(error, hval):
	A = np.zeros((2,2))
	b = np.zeros(2)
	A[0][0] = len(error)
	A[0][1] = sum(np.log(hval)) 
		
	A[1][0] = sum(np.log(hval))
	A[1][1] = sum(np.log(hval)**2)

	b[1] = sum(np.log(hval)*np.log(error))
	b[0] = sum(np.log(error))
	logC, alpha = np.linalg.solve(A,b)
	return alpha



def left(x,on_boundary):
	return near(x[0], 0) and on_boundary

def right(x, on_boundary):
	return near(x[0], 1) and on_boundary

mu1 = [1., 0.1, 0.01, 0.0015, 0.001, 0.0001]
h1 = [8, 16, 32, 64]
P = 1 

PrintTexTable = False
task_d = True


if PrintTexTable == True:
	print '$\\mu$ & h  &   L2  &  H1  & L2 $\\alpha$ & H1 $\\alpha$\\\\ \hline'
#for h in h1:

for mu in mu1:
	t = 0
	H1_error = []
	L2_error = []
	hval = []
	for h in h1:
		
		mesh = UnitSquareMesh(h, h)

		beta = 0.5*mesh.hmin()

		V = FunctionSpace(mesh, 'CG', P)
		V2 = FunctionSpace(mesh, 'CG', P+2)

		bc = [DirichletBC(V, Constant(0), left), DirichletBC(V, Constant(1), right)]

		u = TrialFunction(V)
		v = TestFunction(V)
		L = v + beta * v.dx(0) #SUPG Testfunction
		
		f = Constant(0)

		u_ = Function(V)

		if task_d == True:
			FSD =  mu*inner(grad(u), grad(L))*dx + u.dx(0)*L*dx -f*L*dx
			solve(lhs(FSD) == rhs(FSD), u_, bc)
		else:
			F =  mu*inner(grad(u), grad(v))*dx + u.dx(0)*v*dx -f*v*dx
			solve(lhs(F) == rhs(F), u_, bc)

		u_ex = Expression('(1-exp(x[0]/mu))/(1-exp(1/mu))', mu = mu)
		u_e = interpolate(u_ex, V2)
		
		ud= abs(u_ - u_e)
		#print ud

		L2_norm = errornorm(u_, u_e)
		H1_norm= errornorm(u_, u_e, 'H1')

		
		H1_error.append(H1_norm)
		L2_error.append(L2_norm)
		hval.append(mesh.hmin())
		
		if PrintTexTable == True:
			if t==0:
				print '\multirow{4}{*} {%s} &' %mu,h,'&',L2_norm,' &',H1_norm,' & \multirow{6}{*} {L2} & \multirow{4}{*} {H1} \\\\ \cline{2-4}' 
				t+= 1
			else:
				print '&',h,'&', L2_norm,'&', H1_norm, '&', '&', '\\\\ \cline{2-4}'

		#plot(u_, interactive=True)
	if PrintTexTable == True:
		print '  \hline \hline'

	else:
		
		L2alpha = alpha(L2_error, hval)
		H1alpha = alpha(H1_error, hval)
		print 'mu:', mu, '    L2alpha:', L2alpha, '     H1alpha:', H1alpha
		
		'''
		print mu
		for n in range(len(h1)):
			if n>0:
				print 'L2:', np.log(L2_error[n]/L2_error[n-1]) / np.log(hval[n]/hval[n-1])
				print 'H1:',np.log(H1_error[n]/H1_error[n-1]) / np.log(hval[n]/hval[n-1])
		'''



#plot(ud, interactive=True)	



