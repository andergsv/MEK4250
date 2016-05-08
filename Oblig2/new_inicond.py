from dolfin import *
import numpy as np 
import matplotlib.pyplot as plt
set_log_active(False)

L = 6. #Lengde x-retning
Br = 2. #Lengde y-retning
def functionspace(N):
	for n in range(len(N)):

		dt = 0.01
		T = 5

		A = 1. #Constant(1)
		mesh = RectangleMesh(Point(0,0), Point(L,Br), N[n], N[n])
		#mesh = UnitSquareMesh(N[n], N[n])

		V = FunctionSpace(mesh, 'CG', 1)
		Q = FunctionSpace(mesh, 'CG', 4)

		l= L-Br
		class MyExpression1(Expression):
			def eval_cell(self, value, x, ufc_cell):
				if x[0] < l:
					value[0] = 2*A*cos(pi*x[0]/(2*l))*cos(pi*x[0]/(2*l))
				else:
					value[0] = 0.

		g = MyExpression1(degree=1)
		#g = Expression('A*cos(w*t+delta)*cos(kx*x[0])*cos(ky*x[1])', A = A, delta=delta, t=0, w=np.sqrt(w2), kx=kxc, ky=kyc)
		
		eta0 = project(g,V)
		phi0 = project(Constant(0),V)

		mu = Constant(1.) 

		phi = TrialFunction(V)
		eta = TrialFunction(V)
		Ni = TestFunction(V)

		phi_ = Function(V)
		eta_ = Function(V)

		dtc =Constant(dt)
		F1 = (phi-phi0)/dtc * Ni * dx + eta0 * Ni * dx + mu*mu/Constant(3) * inner(grad(Ni),grad((phi-phi0)/dtc))*dx 
		F2 = (eta-eta0)/dtc * Ni * dx - inner(grad(Ni), grad(phi0)) * dx 

		t = dt
		b = 0

		while t< T:
			solve(lhs(F1)==rhs(F1), phi_)
			phi0.assign(phi_)

			solve(lhs(F2)==rhs(F2), eta_)
			eta0.assign(eta_)
			b += 1
			if b == 100:
				phiH1_norm = np.sqrt(assemble(phi_*phi_*dx + inner(grad(phi_),grad(phi_))*dx))
				etaH1_norm = np.sqrt(assemble(eta_*eta_*dx + inner(grad(eta_),grad(eta_))*dx))
				phiL2_norm = np.sqrt(assemble(phi_**2*dx))
				etaL2_norm = np.sqrt(assemble(eta_**2*dx))
				print '%1.5f %7.5f %7.5f %7.5f %7.5f' %(mesh.hmin(), phiL2_norm, phiH1_norm, etaL2_norm, etaH1_norm)
			plot(eta_, rescale=False)#, interactive=True)
			t +=dt
		
N = [2, 4, 8, 16, 32, 64]# 64, 128]

functionspace(N)