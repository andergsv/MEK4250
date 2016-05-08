from dolfin import *
import numpy as np 
import matplotlib.pyplot as plt
set_log_active(False)

L = 5. #Lengde x-retning
Br = 1. #Lengde y-retning
def functionspace(N, kx, ky):
	a = 0
	print 'numerical error for:'
	for i in kx:
		for j in ky:
			print 'Kx:', i, 'ky:', j
			errorph= np.zeros(len(N))
			erroret=  np.zeros(len(N))
			herrorph= np.zeros(len(N))
			herroret=  np.zeros(len(N))
			h = np.zeros(len(N))
			NN = np.zeros(len(N))
			for n in range(len(N)):

				dt = 0.01
				T = 5

				eps = 1
				A = 1. 
				#mesh = RectangleMesh(Point(0,0), Point(L,Br), N[n], N[n])
				mesh = UnitSquareMesh(N[n], N[n])

				V = FunctionSpace(mesh, 'CG', 1)
				Q = FunctionSpace(mesh, 'CG', 4)

				mu = 1
				kxc = i*np.pi
				kyc = j*np.pi
				w2 = (kxc**2+kyc**2)/(1+mu**2*(kxc**2+kyc**2)/3.)
				B = -float(A)*np.sqrt(w2)/(kxc**2+kyc**2)
				delta = 0


				g = Expression('A*cos(w*t+delta)*cos(kx*x[0])*cos(ky*x[1])', A = A, delta=delta, t=0, w=np.sqrt(w2), kx=kxc, ky=kyc)
				r = Expression('B*sin(w*t+delta)*cos(kx*x[0])*cos(ky*x[1])', B = B, delta=delta, t=0, w=np.sqrt(w2), kx=kxc, ky=kyc)


				eta0 = project(g,V)
				phi0 = project(r,V)

				mu = Constant(mu) 

				phi = TrialFunction(V)
				eta = TrialFunction(V)
				Ni = TestFunction(V)

				phi_ = Function(V)
				eta_ = Function(V)

				dtc =Constant(dt)
				F1 = (phi-phi0)/dtc * Ni * dx + eta0 * Ni * dx + mu*mu/Constant(3) * inner(grad(Ni),grad((phi-phi0)/dtc))*dx 
				F2 = (eta-eta0)/dtc * Ni * dx - inner(grad(Ni), grad(phi0)) * dx 

				#A1 = assemble(lhs(F1))
				#A2 = assemble(lhs(F2))

				t = dt
				etae = Expression('A*cos(w*t+delta)*cos(kx*x[0])*cos(ky*x[1])', A = A, delta=delta, t=0+dt/2., w=np.sqrt(w2), kx=kxc, ky=kyc)
				phie = Expression('B*sin(w*t+delta)*cos(kx*x[0])*cos(ky*x[1])', B = B, delta=delta, t=0, w=np.sqrt(w2), kx=kxc, ky=kyc)

				while t< T:

					#b1 = assemble(rhs(F1))
					#b2 = assemble(rhs(F2))

					etae.t = t+dt/2.
					phie.t = t

					#solve(A1,phi_.vector(), b1)
					solve(lhs(F1)==rhs(F1), phi_)
					phi0.assign(phi_)

					#solve(A2,eta_.vector(), b2)
					solve(lhs(F2)==rhs(F2), eta_)
					eta0.assign(eta_)
						
					#plot(eta_, rescale=False)#, interactive=True)
					t +=dt

				etah = project(etae, Q)
				phih = project(phie, Q)

				errorph[n] = errornorm(eta_, etah, 'l2')
				erroret[n] = errornorm(phi_, phih, 'l2')

				herrorph[n] = errornorm(eta_, etah, 'h1')
				herroret[n] = errornorm(phi_, phih, 'h1')
				NN[n] = N[n]
				h[n] = 1./N[n] #mesh.hmin() #
				
				
				if n == 0:
					print 'n=', N[n], '       l2 phi error   %.5g      l2 eta error     %.5g      h1 phi error    %.5g          h1 eta error  %.5g' %(errorph[n], erroret[n], herrorph[n], herroret[n])
					
				else:
					convergenceph = np.log(abs(errorph[n]/errorph[n-1]))/np.log(abs(h[n]/h[n-1]))
					convergenceet = np.log(abs(erroret[n]/erroret[n-1]))/np.log(abs(h[n]/h[n-1]))
					hconvergenceph = np.log(abs(herrorph[n]/herrorph[n-1]))/np.log(abs(h[n]/h[n-1]))
					hconvergenceet = np.log(abs(herroret[n]/herroret[n-1]))/np.log(abs(h[n]/h[n-1]))
					print 'n=', N[n],  '       l2 phi error   %.5g      l2 eta error     %.5g      l2 phi conv     %0.5f         l2 eta conv   %0.5f' %(errorph[n], erroret[n], convergenceph, convergenceet)
					print '            h1 phi error   %.5g     h1 eta error     %.5g       h1 phi conv     %0.5f         h1 eta conv   %0.5f' %(herrorph[n], herroret[n], hconvergenceph, hconvergenceet)
			
			#wiz1 = plot(eta_, interactive=False)
			#wiz2 = plot(etah, interactive=False)
			#wiz3 = plot(phi_, interactive=False)
			#wiz4 = plot(phih, interactive=False)

			#wiz1.write_png('etanum%s%s'%(i,j))
			#wiz2.write_png('etaex%s%s'%(i,j))
			#wiz3.write_png('phinum%s%s'%(i,j))
			#wiz4.write_png('phiex%s%s'%(i,j))
			a+=2
			plt.figure(a-1)	
			plt.plot(NN, errorph, label='phi L2-error')#'phi error k = [%s pi, %s pi]' %(i, j))
			plt.plot(NN, erroret, label='eta L2-error')#'eta error k = [%s pi, %s pi]' %(i, j))
			plt.plot(NN, herrorph, label='phi H1-error')#'phi error k = [%s pi, %s pi]' %(i, j))
			plt.plot(NN, herroret, label='eta H1-error')#'eta error k = [%s pi, %s pi]' %(i, j))
			plt.legend(loc=1)
			plt.savefig('fig/l2error%s%s.png'%(i,j))

			plt.figure(a)
			plt.loglog(h, errorph, label='phi L2-error')#'phi error k = [%s pi, %s pi]' %(i, j))
			plt.loglog(h, erroret, label='eta L2-error')#'eta error k = [%s pi, %s pi]' %(i, j))
			plt.loglog(h, herrorph, label='phi H1-error')#'phi error k = [%s pi, %s pi]' %(i, j))
			plt.loglog(h, herroret, label='eta H1-error')#'eta error k = [%s pi, %s pi]' %(i, j))
			plt.legend(loc=4)
			plt.savefig('fig/loglog%s%s.png'%(i,j))
			
			#plt.show()

N = [2, 4, 8, 16, 32, 64]# 64, 128]
kx = [1, 2]
ky = [0, 1]

functionspace(N, kx, ky)