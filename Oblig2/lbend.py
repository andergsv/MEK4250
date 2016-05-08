from dolfin import *
import numpy as np 
import matplotlib.pyplot as plt
import meshgenerator 
import os 
set_log_active(False)

L = 6. #Lengde x-retning
B = 2. #Lengde y-retning

def Lmesh(L,B,grading, name, create_mesh=False):
	if create_mesh==True:
		meshgenerator.L(name+'.geo', L, B, grading) 
		os.system('gmsh -2 '+name+'.geo')
		os.system('dolfin-convert '+ name+'.msh '+name+'.xml')

Lmesh(L,B,1, 'test', False)
mesh = Mesh('test.xml')#mesh1 + RectangleMesh(Point(L-B,B), Point(L, 20), ny, nx)

wiz1 = plot(mesh, interactive=False)
wiz1.write_png('meshL')
def functionspace(mesh, N):
	#print 'mesh size       phiL2       phiH1       etaL2       phiH1'
	for n in range(N):
		if n == 0:
			None
		else:
			mesh = refine(mesh)

		dt = 0.01
		T = 10

		A = 1.

		V = FunctionSpace(mesh, 'CG', 1)

		l= L-B
		class MyExpression1(Expression):
			def eval_cell(self, value, x, ufc_cell):
				if x[0] < l:
					value[0] = 2*A*cos(pi*x[0]/(2*l))*cos(pi*x[0]/(2*l))
				else:
					value[0] = 0.

		g = MyExpression1(degree=1)
		
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
		a = 0
		b = 0
		while t< T:

			solve(lhs(F1)==rhs(F1), phi_)
			phi0.assign(phi_)

			solve(lhs(F2)==rhs(F2), eta_)
			eta0.assign(eta_)
				
			#plot(eta_, rescale=False)#, interactive=True)
			
			a += 1
			#b += 1
			#if b == 100:

				#phiH1_norm = np.sqrt(assemble(phi_*phi_*dx + inner(grad(phi_),grad(phi_))*dx))
				#etaH1_norm = np.sqrt(assemble(eta_*eta_*dx + inner(grad(eta_),grad(eta_))*dx))
				#phiL2_norm = np.sqrt(assemble(phi_**2*dx))
				#etaL2_norm = np.sqrt(assemble(eta_**2*dx))
				#print '%1.5f %7.5f %7.5f %7.5f %7.5f' %(mesh.hmin(), phiL2_norm, phiH1_norm, etaL2_norm, etaH1_norm)
			if a == 100:
				wiz = plot(eta_, interactive=False)
				wiz.write_png('timeeta%s%s' %(n,int(t)))
				a = 0
			t +=dt

N = 5

functionspace(mesh,N)