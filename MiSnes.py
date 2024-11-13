import dolfinx 
import numpy as np
import ufl

import os, shutil, sys
def LINE():
    #return sys._getframe(2).f_lineno
    return f"{sys._getframe(2).f_code.co_filename}:{sys._getframe(2).f_lineno:04d}"
def Lprint(uno):
    #return print(f" test6.3.py:{LINE()} :",uno)
    #Lprint(f"A={A},B={B},...")
    return print(f"{LINE()} :",uno)


class JSMSnes():
    
    def __init__(self, V, a, L, bc,y0=0):
        self.AA=dolfinx.fem.assemble_matrix(dolfinx.fem.form(a), bcs=bc)
        self.bb = dolfinx.fem.assemble_vector(dolfinx.fem.form(L))
        dolfinx.fem.apply_lifting(self.bb.array, [dolfinx.fem.form(a)], bcs=[bc])

        #LD_min = np.min(self.bb.array)
        #LD_max = np.max(self.bb.array)

        #print(f'LD_min={LD_min}, LD_max={LD_max}')
        if False:
            self.xx = V.tabulate_dof_coordinates()
            self.x_order0 = np.argsort(self.xx[:,0])
            self.x_order=[]
            for i in self.x_order0:
    #            Lprint(f"self.xx[i]={self.xx[i]}")
                if self.xx[i][1]==y0:
                    self.x_order.append(i)
        self.xx = V.tabulate_dof_coordinates()
        self.x_order0 = np.argsort(self.xx[:,2])
        self.x_order=[]
        for i in self.x_order0:
#            Lprint(f"self.xx[i]={self.xx[i]}")
            if np.abs(self.xx[i][0])<0.001 and np.abs(self.xx[i][1])<0.001:
                self.x_order.append(i)
#        Lprint(f"self.x_order={self.x_order}")

    def Jacobi_iteracion(self,uh,boundary_dofs):
            ###########################################
        #### iteracion del Metodo de Jacobi Proyectado ##########
        ###########################################
        self.err=0
        for i in range(len(uh.x.array)):
            
#            if (fD.x.array[i]<=0) | (i in boundary_dofs):
            if (self.bb.array[i]<=0) | (i in boundary_dofs):

                
                tmp=0
            else:
                

                fila=self.AA.data[self.AA.indptr[i]:self.AA.indptr[i+1]]
                cols=self.AA.indices[self.AA.indptr[i]:self.AA.indptr[i+1]]
                tmp=self.bb.array[i]
                #Lprint(f"tmp={tmp}")
                for k in range(len(fila)):
                    icol=cols[k]
                    if icol==i:
                        aii=fila[k]
                #       Lprint(f"aii={aii}")
                    else:
                        tmp-=fila[k]*uh.x.array[icol]

                tmp=tmp/aii
                
                if tmp>0.95:
                    tmp=0.95
                #LO AGREGUE PARA NO CONSIDERAR SOLUCIONES NEGATIVAS
                if tmp<0.:
                    tmp=0.
                
            errtmp=abs(uh.x.array[i]-tmp)       
            if errtmp>self.err:
                self.err=errtmp 
            uh.x.array[i]=tmp
            
        return uh

    def Jacobi_Bounded(self,uh,uh0,fD,boundary_dofs,i_t):
        """uh solo se usa para graficar \n 
        uh0 se calcula con Jacobi"""
        import matplotlib.pyplot as plt
        import time
        for itJ in range(401):
            
            self.Jacobi_iteracion(uh0,boundary_dofs)
            if (itJ %5==0 ):
                print(f'Jacobi Iteration: {itJ}, err={self.err:.3e}')
            if False:
                if (itJ %20==0 ):
                    
                    time.sleep(.1)
                    plt.clf()
                    plt.plot(self.xx[self.x_order, 0], uh .x.array[self.x_order]   ,color='blue', linewidth=1)
                    plt.plot(self.xx[self.x_order, 0], uh0.x.array[self.x_order]   ,color='blue', linewidth=2)
                    plt.plot(self.xx[self.x_order, 0], fD .x.array[self.x_order]/40,color='red', linewidth=1)
                    
                    plt.legend(['sol $-\Delta u =f$',
                                'sol Jacobi',
                                'Lado derecho f'], loc="upper left") 

                    y=max(uh.x.array)

                    plt.text(.62,y,f"itJ={itJ}, err={self.err:.3e}")
                    plt.grid()
                    plt.savefig("u_1d.png")
            Testigo = 1
            if Testigo == 1:
                if (itJ %5==0 ):
                    
                    time.sleep(.1)
                    plt.clf()
                    plt.plot(self.xx[self.x_order, 2], uh .x.array[self.x_order]   ,color='blue', linewidth=1)
                    plt.plot(self.xx[self.x_order, 2], uh0.x.array[self.x_order]   ,color='blue', linewidth=2)
                    plt.plot(self.xx[self.x_order, 2], fD .x.array[self.x_order]/40,color='red', linewidth=1)
                    
                    plt.legend(['sol Laplaciano',
                                'sol Jacobi',
                                'Lado derecho'], loc="upper left") 

                    y=max(uh.x.array)

                    plt.text(.62,y,f"itJ={itJ}, err={self.err:.3e}")
                    plt.grid()
                    plt.savefig(f"sol_Jacobi_{i_t}.png")
            if self.err<1e-3:#self.err<1e-5:
                break
    
        return uh0

