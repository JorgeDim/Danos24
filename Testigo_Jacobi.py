'''
Authors: 
    CMM-Mining                                                             
Purpose:                                              
    This script simulates fracture mechanics via a coupled linear
    elasticity - gradient damage model. The total energy is computed
    and thus a variational formulation is then derived.                                                                             
'''
# ========================================
# Import libraries to get the code working
# ========================================
import matplotlib.pyplot as plt
import numpy as np
import sympy 
import socket
import datetime

import dolfinx
import dolfinx.plot
import dolfinx.io 
import dolfinx.cpp
import dolfinx.fem.forms
import dolfinx.fem.function
import dolfinx.fem.petsc
import ufl

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from dolfinx import default_scalar_type

import os, shutil, sys

from Utils.snes_problem import SNESProblem

import MiSnes



def LINE():
    #return sys._getframe(2).f_lineno
    return f"{sys._getframe(2).f_code.co_filename}:{sys._getframe(2).f_lineno}"
def Lprint(uno):
    #return print(f" test6.3.py:{LINE()} :",uno)
    #Lprint(f"A={A},B={B},...")
    return print(f"{LINE()} :",uno)

if MPI.COMM_WORLD.rank == 0:
    print("============================================")
    print(f"This code is built in DOLFINx version: {dolfinx.__version__}")
    print("============================================")

# =============================
# Read mesh from external files
# =============================
if MPI.COMM_WORLD.rank == 0:
    print("=============================")
    print("Read mesh from external files")
    print("=============================")
#malla = "Meshes/cilinder-75_300_150.xdmf"
#malla = "Meshes/mesh_normal.xdmf"
malla = "Meshes/malla.xdmf"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, malla, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

# geometry dimension
ndim    = mesh.geometry.dim
fdim    = mesh.topology.dim-1
n       = ufl.FacetNormal(mesh)

# =====================================================
# Material constant. The values of parameters involved
# in the models are defined here
# =====================================================
#E, nu   = dolfinx.fem.Constant(mesh, 2.5e10), dolfinx.fem.Constant(mesh, 0.3)
E, nu   = dolfinx.fem.Constant(mesh, 2.5e10), dolfinx.fem.Constant(mesh, 0.25)
kappa   = dolfinx.fem.Constant(mesh,0.0)
#Gc      = dolfinx.fem.Constant(mesh, 1.0e5)
ell     = dolfinx.fem.Constant(mesh, 0.01) #dolfinx.fem.Constant(mesh, 0.8944)
f       = dolfinx.fem.Constant(mesh,np.array([0,0,0], dtype=np.float64))
a_alpha = "quad"
w_alpha = "quad"

relax=1.0

Largo           = np.max(mesh.geometry.x[:,2])-np.min(mesh.geometry.x[:,2])
cargacontrolada = False
condicioninicial = False
# Parameter for minimization
alt_min_parameters = {"atol": 1.e-5,"max_iter": 100}

# ========================
# Create Results Directory
# ======================== 
ahora=datetime.datetime.now().strftime("%y-%m")+'_'+socket.gethostname()
modelname   = "[CargaCont=%s]_[ConInic=%s]_[a_alpha=%s]_[w_alpha=%s]_[kappa=%1.2f]_[relax=%1.1f]"%(cargacontrolada,condicioninicial,a_alpha,w_alpha,kappa.value,relax)
if MPI.COMM_WORLD.rank == 0:
    print(modelname)
#savedir     = "results/testigo_%s/%s"%(ahora,modelname)
savedir     = "results/testigo_mallanormal_%s/%s"%(ahora,modelname)
if os.path.isdir(savedir):
    shutil.rmtree(savedir)
# ================================================
# Create function space for 3D elasticity + Damage
# ================================================
# Define the function space for displacement
V_u         = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, )))
# Define the function space for damage
V_alpha         = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
# Define the state
u           = dolfinx.fem.Function(V_u, name="Displacement")
alpha       = dolfinx.fem.Function(V_alpha, name="Damage")
alpha0     = dolfinx.fem.Function(V_alpha, name="Damage")
alphaJ     = dolfinx.fem.Function(V_alpha, name="Damage")
Reaccion   = dolfinx.fem.Function(alpha.function_space)
alpha_dot   = dolfinx.fem.Function(V_alpha, name="Derivative Damage")
state       = {"u": u, "alpha": alpha}
# need upper/lower bound for the damage field
alpha_lb    =   dolfinx.fem.Function(V_alpha, name="Lower bound")
alpha_ub    = dolfinx.fem.Function(V_alpha, name="Upper bound")
# Measure
dx  = ufl.Measure("dx",domain=mesh)

# ======================================================
# Boudary conditions:
# Brief description about the set of boundary conditions 
# for the displacement field.
# ======================================================
def bottom(x):
    return np.isclose(x[2], np.min(x[2]))
def top(x):
    return np.isclose(x[2], np.max(x[2]))
def lateral(x):
    radio2 = np.max(x[0]**2+x[1]**2)**2
    return np.isclose(x[0]**2+x[1]**2, radio2)
# Boundary facets and dofs for displacement
boundary_facets_top_u   = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim-1, top)

blocked_dofs_top_u      = dolfinx.fem.locate_dofs_topological(V_u.sub(2), mesh.topology.dim-1, boundary_facets_top_u)
blocked_dofs_bottom_u   = dolfinx.fem.locate_dofs_geometrical(V_u, bottom)
# Boundary dofs for damage
blocked_dofs_top_alpha = dolfinx.fem.locate_dofs_geometrical(V_alpha, top)
blocked_dofs_bottom_alpha = dolfinx.fem.locate_dofs_geometrical(V_alpha, bottom)
blocked_dofs_lateral_alpha = dolfinx.fem.locate_dofs_geometrical(V_alpha, lateral)
# Define values of boundary condition for displacement or load
# Vector value for zero displacement
zero_u = dolfinx.fem.Constant(mesh, np.zeros(mesh.geometry.dim, dtype=default_scalar_type))
#with zero_u.vector.localForm() as bc_local:
#    bc_local.set(0.0)
#zero_u.vector.destroy()
# Scalar value for non-zero load
nonzero_load = dolfinx.fem.Constant(mesh, ScalarType(1.0))
# Scalar value for non-zero damage


zero_alpha = dolfinx.fem.Constant(mesh, 0.0)
one_alpha = dolfinx.fem.Constant(mesh, 1.0)



# Define the Dirichlet boundary conditions for displacement
nonzero_u = dolfinx.fem.Constant(mesh, ScalarType(1.0))
zero_u_escalar = dolfinx.fem.Constant(mesh, ScalarType(0.0))
bc_u0 = dolfinx.fem.dirichletbc(zero_u, blocked_dofs_bottom_u,V_u)
bc_u1 = dolfinx.fem.dirichletbc(nonzero_u, blocked_dofs_top_u, V_u.sub(2))
bc_u2 = dolfinx.fem.dirichletbc(zero_u_escalar, blocked_dofs_top_u, V_u.sub(1))
bc_u3 = dolfinx.fem.dirichletbc(zero_u_escalar, blocked_dofs_top_u, V_u.sub(0))
# Define the Dirichlet boundary conditions for damage
bc_alpha0 = dolfinx.fem.dirichletbc(zero_alpha, blocked_dofs_top_alpha,V_alpha)
bc_alpha1 = dolfinx.fem.dirichletbc(zero_alpha, blocked_dofs_bottom_alpha,V_alpha)
bc_alpha2 = dolfinx.fem.dirichletbc(zero_alpha, blocked_dofs_lateral_alpha,V_alpha)
# Merge the boundary condition for displacement and damage
if cargacontrolada:
    bcs_u = [bc_u0]
else:
    bcs_u = [bc_u0,bc_u1]

bc_alpha_old =[]
bcs_alpha = [bc_alpha0,bc_alpha1]
# setting the upper bound to 0 where BCs are applied

alpha_ub.x.array[:] = one_alpha
#alpha_ub.interpolate(one_alpha)

#dolfinx.fem.set_bc(alpha_ub.vector, bcs_alpha)
# Define Neumann bondary conditions
# Set markers and locations of boundaries
if cargacontrolada:
    boundaries = [(1,top),(2,bottom),(3,lateral)]
    facet_indices, facet_markers = [],[]
    for (marker, locator) in boundaries:
        facets =  dolfinx.mesh.locate_entities_boundary(mesh,fdim,locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets,marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag =  dolfinx.mesh.meshtags(mesh,fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    # Define new measure including boundary naming
    # top: ds(1), bottom:  ds(2)
    # lateral : ds(3)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)
else:
    ds=ufl.Measure("ds", domain=mesh)
def initial_condition(x):
     a_1=0.002
     a_2=0.002
     a_3=0.00001
     term_1 = ((np.cos(np.pi/4)*(x[0])+np.sin(np.pi/4)*(x[2]))**2)/a_1
     term_2 = (x[1]**2)/a_2
     term_3 = ((-np.sin(np.pi/4)*(x[0])+np.cos(np.pi/4)*(x[2]))**2)/a_3
     if condicioninicial:
        return 1.0*((term_1+term_2+term_3)<=1)
     else:
         return 0.0*((term_1+term_2+term_3)<=1)
# =====================================================
# In this block of code define the operators.  These is
# independent from the mesh.
# -----------------------------------------------------
# Constitutive functions of the damage model. Here
# we define the operators acting on the damage system
# as well as the operator acting on the displacement
# field, which depends on the damage.
# =====================================================
C_1=-0.31
C_2=1.9448411e5/60/10
C_3=0*-198.94/60/10  #acá debe ser negativo

def w(alpha):
    """Dissipated energy function as a function of the damage """
    return C_2*alpha + C_3*alpha**2
def Dw_Dalpha(alpha):
    """Dissipated energy function as a function of the damage """
    return C_2 + 2*C_3*alpha
def a(alpha, k_ell=1.e-6):
    """Stiffness modulation as a function of the damage """
    return 1-(1+C_1)*alpha+C_1*alpha**2
def Da_Dalpha(alpha, k_ell=1.e-6):
    """Stiffness modulation as a function of the damage """
    return -(1+C_1)+2*C_1*alpha
def eps(u):
    """Strain tensor as a function of the displacement"""
    return ufl.sym(ufl.grad(u))
def sigma_0(u):
    """Stress tensor of the undamaged material as a function of the displacement"""
    mu    = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu)*(1-2*nu)) 
    return 2.0 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(ndim)
def sigma(u,alpha):
    """Stress tensor of the damaged material as a function of the displacement and the damage"""
    return a(alpha) * sigma_0(u)
def Dsigma_Dalpha(u,alpha):
    """Stress tensor of the damaged material as a function of the displacement and the damage"""
    return Da_Dalpha(alpha) * sigma_0(u)
#  Define th Deviator and Spheric Tensors
def Dev(Tensor):
    return Tensor-Sph(Tensor)
def Sph(Tensor):  
    #Tensor2=ufl.Identity(ndim) 
    Tensor2=ufl.as_matrix([[nu/(1-nu),0,0],[0,nu/(1-nu),0],[0,0,1]])
    return (ufl.inner(Tensor,Tensor2)/ufl.inner(Tensor2,Tensor2))*Tensor2 

def LadoDerecho_alpha():
    Denergia_Dalpha=dolfinx.fem.Function(V_alpha ,name="energia_alpha")

    a_tmp = ufl.inner(ufl.TrialFunction(V_alpha), ufl.TestFunction(V_alpha)) * ufl.dx
    L_tmp = ufl.inner(Ddensidad_energy2_Dalpha, ufl.TestFunction(V_alpha)) * ufl.dx
    problem_tmp = dolfinx.fem.petsc.LinearProblem(a_tmp, L_tmp)
    Denergia_Dalpha = problem_tmp.solve()
    return Denergia_Dalpha

# =====================================================
# Constants
# =====================================================
z       = sympy.Symbol("z")
c_w     = 4*sympy.integrate(sympy.sqrt(w(z)),(z,0,1))
Gc      = dolfinx.fem.Constant(mesh, float(c_w)*3577)
#print("c_w = ",c_w)
c_1w    = sympy.integrate(sympy.sqrt(1/w(z)),(z,0,1))
#print("c_1/w = ",c_1w)
tmp     = 2*(sympy.diff(w(z),z)/sympy.diff(1/a(z),z)).subs({"z":0})
sigma_c = sympy.sqrt(tmp * Gc.value * E.value / (c_w * ell.value))
#print("sigma_c = %2.3f"%sigma_c)
eps_c   = float(sigma_c/E.value)
#print("eps_c = %2.3f"%eps_c)

# =====================================================
# Useful functions for minimization
# =====================================================
def simple_monitor(state, iteration, error_L2):
    alpha       = state["alpha"]
    alpha_max   = np.amax(alpha.x.array)
    if MPI.COMM_WORLD.rank == 0:
        print(f"Iteration: {iteration:3d}, Error: {error_L2:3.4e}, Max Alpha: {alpha_max:.3f}")
def alternate_minimization(state,problem_u,solver_alpha_snes,parameters=alt_min_parameters,monitor=None):
    u           = state["u"]
    alpha       = state["alpha"]
    alpha_old   = dolfinx.fem.Function(alpha.function_space)
    # Set previous alpha as alpha_old
    alpha_old.x.array[:]=alpha.x.array
   #alpha.vector.copy(alpha_old.vector)
    for iteration in range(parameters["max_iter"]):                 
        # solve displacement
        problem_u.solve()

        #u_file.write_function(u,t)
        ###########
        ###########
        # SOLVE DAMAGE JACOBI
        ###########
        ###########
        LDA=LadoDerecho_alpha()
        #LDA_file.write_function(LDA,t)
        lista_boundary_dofs=[]
        lista_boundary_dofs1=[]
        for i in range(len(LDA.x.array)):
            if LDA.x.array[i]>=0 & (not i in lista_boundary_dofs):
                lista_boundary_dofs.append(i)
        boundary_dofs=np.array(lista_boundary_dofs).astype(int)
        bc_alpha_old = dolfinx.fem.dirichletbc(alpha_lb, boundary_dofs)
        #boundary_dofs1=np.array(lista_boundary_dofs1).astype(int)
        
        #problem_alpha   = dolfinx.fem.petsc.LinearProblem(a=ufl.lhs(E_dalpha ), L=ufl.rhs(E_dalpha), bcs=[bc_alpha_old,bc_alpha0,bc_alpha1], u=alpha,
        #                                            petsc_options={"ksp_type": "gmres", "pc_type": "none","ksp_rtol": "1e-10"})
        problem_alpha   = dolfinx.fem.petsc.LinearProblem(a=ufl.lhs(E_dalpha ), L=ufl.rhs(E_dalpha), bcs=[bc_alpha_old,bc_alpha0,bc_alpha1], u=alpha)
        problem_alpha.solve()
        #solver_alpha_snes.solve(None, alpha.vector)

        #alpha_file.write_function(alpha,t)
        alpha0.x.array[:]=alpha.x.array
        alphaJ.x.array[:]=alpha.x.array
        problemJ=MiSnes.JSMSnes( V_alpha, ufl.lhs(E_dalpha ), ufl.rhs(E_dalpha ), [bc_alpha_old,bc_alpha0,bc_alpha1])
        #problemJ=MiSnes.JSMSnes( V_alpha, ufl.lhs(E_dalpha ), ufl.rhs(E_dalpha ), bcs_alpha)
        
        problemJ.Jacobi_Bounded(alpha0,alphaJ,LDA,boundary_dofs,i_t)
        #alphaJ_file.write_function(alphaJ,t)
        
        alpha.x.array[:]=alpha_old.x.array+relax*(alphaJ.x.array-alpha_old.x.array)

        # check error and update
        L2_error    = dolfinx.fem.form(ufl.inner(alpha - alpha_old, alpha - alpha_old) * dx)
        error_L2    = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(L2_error),op=MPI.SUM))
        # uptade alpha_old with the value of alpha
        alpha_old.x.array[:]=alpha.x.array
        
        #alpha.vector.copy(alpha_old.vector)
        # Monitor of solutions
        if monitor is not None:
            monitor(state, iteration, error_L2)
        # check error                      
        if error_L2 <= parameters["atol"]:
            break
    else:
        pass #raise RuntimeError(f"Could not converge after {iteration:3d} iteration, error {error_L2:3.4e}") 
    return (error_L2, iteration)
def postprocessing(state, iteration, error_L2):
    # Save number of iterations for the time step
    u = state["u"]
    alpha = state["alpha"]
    iterations[i_t] = np.array([t,i_t])
    # Compute integrals
    vol             = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * dx))
    #Largo           = np.max(x[2])-np.min(x[2])
    int_alpha_temp  = alpha * dx
    int_alpha       = dolfinx.fem.assemble_scalar(dolfinx.fem.form(int_alpha_temp))
    alpha_max       = np.amax(alpha.x.array)
    int_sigmazz_temp  = sigma(u,alpha)[2,2]/vol * dx
    int_sigmazz       = dolfinx.fem.assemble_scalar(dolfinx.fem.form(int_sigmazz_temp))
    if cargacontrolada:
        int_epszz_temp  = eps(u)[2,2]/vol * dx
        epsilon_maquina      = dolfinx.fem.assemble_scalar(dolfinx.fem.form(int_epszz_temp))
    else:
        epsilon_maquina = nonzero_u/Largo
    # Compute energies
    elastic_energy_value    = dolfinx.fem.assemble_scalar(dolfinx.fem.form(elastic_energy))
    surface_energy_value    = dolfinx.fem.assemble_scalar(dolfinx.fem.form(dissipated_energy))
    energies[i_t]           = np.array([i_t,elastic_energy_value,surface_energy_value,elastic_energy_value+surface_energy_value])
    ints_domains[i_t]       = np.array([i_t,int_alpha/vol,abs(int_sigmazz),abs(epsilon_maquina),alpha_max])
    simple_monitor(state, iteration, error_L2)

# =====================================================
# Let us define the total energy of the system as the 
# sum of elastic energy, dissipated energy due to the 
# damage and external work due to body forces. 
# =====================================================
alpha_old   = dolfinx.fem.Function(alpha.function_space)
#def funcR(alpha,alpha_old):
#    x=alpha-alpha_old
#    return alpha*Reaccion
elastic_energy      = 0.5 * ufl.inner(sigma(u,alpha), eps(u)) * dx 
densidad_energy2 =   0.5*(ufl.inner(Dev(sigma(u,alpha)),Dev(eps(u))) \
                        +(1-kappa)*ufl.inner(Sph(sigma(u,alpha)),Sph(eps(u))))
elastic_energy2 =   (densidad_energy2)*dx
Ddensidad_energy2_Dalpha =   0.5*(ufl.inner(Dev(Dsigma_Dalpha(u,alpha)),Dev(eps(u))) \
                        +(1-kappa)*ufl.inner(Sph(Dsigma_Dalpha(u,alpha)),Sph(eps(u)))) \
                        + Dw_Dalpha(alpha)
Lprint(f"w(0)={w(0)},  w(1)={w(1)}")
dissipated_energy   = (w(alpha)  + w(1)*ell**2 * ufl.dot(ufl.grad(alpha), ufl.grad(alpha)) )* dx
external_work       = ufl.dot(f, u) * dx
if cargacontrolada: 
    bounbdary_energy      = ufl.dot(nonzero_load*n,u)*ds(1)
    total_energy        = elastic_energy + dissipated_energy - external_work - bounbdary_energy
else:
    total_energy        = elastic_energy + dissipated_energy - external_work 

total_energy2        = elastic_energy2 + dissipated_energy - external_work #- bounbdary_energy

#fD2=  (0.5*ufl.inner(Dev(Dsigma_Dalpha(u,alpha)),Dev(eps(u))) \
#                        +(1-kappa)*ufl.inner(Sph(Dsigma_Dalpha(u,alpha)),Sph(eps(u))) \
#                        + Dw_Dalpha(alpha) +w(alpha) +funcR(alpha,alpha_old) )/w(1)*ell**2

#fD = dolfinx.fem.Function(V_alpha)
#fD.interpolate(lambda x: (-100*np.sin(3*np.pi*x[0])+50*x[0]*(1-x[0])) )
#fD.interpolate(fD2)

# =====================================================
# Weak form of elasticity problem. This is the formal 
# expression for the tangent problem which gives us the 
# equilibrium equations
# =====================================================
E_u             = ufl.derivative(total_energy,u,ufl.TestFunction(V_u))
E_du            = ufl.replace(E_u,{u: ufl.TrialFunction(V_u)})
E_alpha         = ufl.derivative(total_energy2,alpha,ufl.TestFunction(V_alpha ))
Lprint(f"E_alpha={E_alpha}")
E_dalpha        = ufl.replace(E_alpha,{alpha: ufl.TrialFunction(V_alpha)})
Lprint(f"E_dalpha={E_dalpha}")
E_alpha_alpha   = ufl.derivative(E_alpha,alpha,ufl.TrialFunction(V_alpha ))
jacobian        = dolfinx.fem.form(E_alpha_alpha)
residual        = dolfinx.fem.form(E_alpha)
# Displacement problem
problem_u   = dolfinx.fem.petsc.LinearProblem(a=ufl.lhs(E_du), L=ufl.rhs(E_du), bcs=bcs_u, u=u,
                                      petsc_options={"ksp_type": "cg", "pc_type": "none","ksp_rtol": "1e-10"})
# Damage problem
#problem_alpha   = dolfinx.fem.petsc.LinearProblem(a=ufl.lhs(E_dalpha ), L=ufl.rhs(E_dalpha), bcs=[bcs_alpha,bc_alpha_old],
#                                       petsc_options={"ksp_type": "cg", "pc_type": "none","ksp_rtol": "1e-10"})
#problem_alpha   = dolfinx.fem.petsc.LinearProblem(a=ufl.lhs(E_dalpha ), L=ufl.rhs(E_dalpha), bcs=bcs_alpha,u=alpha,
#                                       petsc_options={"ksp_type": "cg", "pc_type": "none","ksp_rtol": "1e-10"})

problem_alpha   = dolfinx.fem.petsc.LinearProblem(a=ufl.lhs(E_dalpha ), L=ufl.rhs(E_dalpha), bcs=bcs_alpha,u=alpha,
                                                  petsc_options={"ksp_type": "gmres", "pc_type": "none","ksp_rtol": "1e-10"})




# reference value for the loading 
load0   = 1.0
loads   = load0*np.linspace(0.,1.,100)
#loads   = load0*np.linspace(0.,1.,200)
# Create array to save some results
energies        = np.zeros((len(loads),4))
iterations      = np.zeros((len(loads),2))
ints_domains    = np.zeros((len(loads),5))
# Set initial condition for damage
#with alpha.vector.localForm() as alpha_local:
#    alpha_local.set(0)
alpha.interpolate(initial_condition)
# Crete the files to store the solutions
u_file      = dolfinx.io.VTKFile(mesh.comm, savedir+"/u.pvd", "w")
alpha_file  = dolfinx.io.VTKFile(mesh.comm, savedir+"/alpha.pvd", "w")
#alphaJ_file  = dolfinx.io.VTKFile(mesh.comm, savedir+"/alphaJ.pvd", "w")
#LDA_file      = dolfinx.io.VTKFile(mesh.comm, savedir+"/LDA.pvd", "w")
u_file.write_mesh(mesh) 
alpha_file.write_mesh(mesh)  
#alphaJ_file.write_mesh(mesh)  
#LDA_file.write_mesh(mesh)  
load = 0
for i_t, t in enumerate(loads):
    
    if cargacontrolada:
        load=(-4.3e7*t)/3
        nonzero_load.value=load 
    else:
        #load=(-0.0004*t)/10-1.6e-5
        load=0.0004*t
        nonzero_u.value=load 
    # update the lower bound
    alpha_lb.x.array[:]=alpha.x.array
    #alpha.vector.copy(alpha_lb.vector)  
    if MPI.COMM_WORLD.rank == 0:  
        print(f"-- Solving for it = {i_t} --")
        print(f"-- Load = {load:2.7e} --")
    # alternate minimization
    alternate_minimization(state,problem_u,problem_alpha,parameters=alt_min_parameters,monitor=postprocessing)
    # save solutions
    u_file.write_function(u,t)
    alpha_file.write_function(alpha,t)
u_file.close()
alpha_file.close()
#alphaJ_file.close()
#LDA_file.close()

# =====================================================
# Plots
# =====================================================
#plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

t = ints_domains[:,3]
data1 = ints_domains[:,2]
data2 = ints_domains[:,4]
fig, ax1 = plt.subplots()
color = 'blue'

ax1.set_xlabel(r'$\vert \varepsilon_{zz} \vert$')
ax1.set_ylabel(r'$\vert \sigma_{zz}(\varepsilon)\vert $', color=color)
ax1.plot(t, data1,'-',color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()

color = 'red'
ax2.set_ylabel(r'$ \alpha_{max}(\varepsilon) $', color=color)
ax2.plot(t, data2,'-', color=color)
ax2.tick_params(axis='y', labelcolor=color)
#plt.gca().invert_xaxis()
plt.grid()
plt.savefig(savedir+"/sigma_eps.png")
plt.show()
plt.clf()


# Plot integrate of damage
p1, = plt.plot(ints_domains[:,3], ints_domains[:,2],'b*-',linewidth=2)
p2, = plt.plot(ints_domains[:,3], 4.3e7*ints_domains[:,4],'r*-',linewidth=2)
plt.legend([p1,p2], ["sigma(epsilon)","alpha_max(epsilon)"])
plt.gca().invert_xaxis()
plt.grid()
plt.xlabel('epsilon')
plt.ylabel('sigma(b),alpha(r)')
#plt.savefig(savedir+"/sigma_eps.png")
plt.clf()

# Plot integrate of damage
p1, = plt.plot(ints_domains[:,0], ints_domains[:,1],'r*-',linewidth=2)
plt.legend([p1], ["Damage_int"])
plt.xlabel('iteration')
plt.ylabel('Damage')
plt.savefig(savedir+"/Damage_int.png")
plt.show()
plt.clf()

# Plot Energies vs Displacement
p1, = plt.plot(energies[:,0], energies[:,1],'b*',linewidth=2)
p2, = plt.plot(energies[:,0], energies[:,2],'r^',linewidth=2)
p3, = plt.plot(energies[:,0], energies[:,3],'ko',linewidth=2)
plt.legend([p1, p2, p3], ["Elastic","Dissipated","Total"])
plt.xlabel('iteration')
plt.ylabel('Energies')
plt.savefig(savedir+"/energies.png")

fDatos      =   open(savedir + '/Parametros.txt', 'w')
buff='A_alpha   =%s\n'%(a_alpha);fDatos.write(buff)     ; print(buff)
buff='w_alpha   =%s\n'%(w_alpha);fDatos.write(buff)     ; print(buff)
buff='kappa     =%f\n'%(kappa);fDatos.write(buff); print (buff)
buff='E         =%1.1e\n'%(E.value);fDatos.write(buff) ; print (buff)
buff='nu        =%f\n'%(nu.value);fDatos.write(buff)     ; print (buff)
buff='ell       =%f\n'%(ell.value);fDatos.write(buff)   ; print (buff)
buff='c_w       =%f\n'%(float(c_w));fDatos.write(buff)     ; print (buff)
buff='G_c       =%f\n'%(Gc);fDatos.write(buff); print (buff)
fDatos.close()
