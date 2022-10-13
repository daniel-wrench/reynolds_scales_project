#!/usr/bin/env python
#### CHANGELOG STARTED TOO LATE
#### 2022/10/09 - Implemented computation of dfn for a sliding window
####              that is smaller than ngrid.
import sys,os,h5py
import numpy as np
from mpi4py import MPI

# Import TurbAn and Py3D codes
for dd in ['AJGAR', 'AJGAR/Py3D', 'AJGAR/TurbAn', 'WorkSpace/PIC-Distfn']:  
   if os.path.exists(os.environ['HOME']+'/'+dd):
      sys.path.insert(0,os.environ['HOME']+'/'+dd+'/')
from TurbAn.Interfaces.Simulations.p3d import p3d
import TurbAn.Analysis.Simulations.AnalysisFunctions as af 

# import paramters for the specific run
from paramfile import *
from NonMaxwellian import *

#Initiate MPI
comm = MPI.COMM_WORLD
size = comm.Get_size() 
rank = comm.Get_rank() 
status = MPI.Status()
if rank == 0:
   if size != pex*pey: sys.exit("Quitting!! Number of processes not equal to pex*pey!")
   t_start=MPI.Wtime()

my_px=rank%pex
my_py=rank//pey

# rounding off issues were giving a lot of trouble. 
# E.g. 24.00000000001 instead of 24.0
# Hence forcing round off here.
P_DX    = round((xmax-x0)/pex,5)
P_DY    = round((ymax-y0)/pey,5)
x0_proc = round(x0+P_DX*my_px,5)
x1_proc = round(x0_proc+P_DX ,5)
y0_proc = round(y0+P_DY*my_py,5) 
y1_proc = round(y0_proc+P_DY ,5)

# Create rc object to load parameters for min-max dictionary
rc=p3d(basedir,'000',dump_num=dump_num,dump_path=dmpdir) 
rc.vars2load(rc.primitives)
rc.loadslice(slicenum)
# size of the box to read particles from
fdx=round(rc.dx*ngrid,5) 
fdy=round(rc.dy*ngrid,5)

# number of grid points to compute dfn on
# depends on sliden, not ngrid. We want overlapping
# estimates of dfn for larger ngrid
nxp=rc.nx//sliden//pex
nyp=rc.ny//sliden//pey

# List of grid points on the processor for which
# to compute the dfn. Depends on sliden, not ngrid
xr=[np.int(i*sliden) for i in range(my_px*nxp,(my_px+1)*nxp)]
yr=[np.int(i*sliden) for i in range(my_py*nyp,(my_py+1)*nyp)]

# create empty arrays to store nonmaxwellianity on 
# master processor
if rank == 0:
   nmi=np.zeros((rc.nx//sliden,rc.ny//sliden))
   nme=np.zeros((rc.nx//sliden,rc.ny//sliden))

# Temporary working arrays for each processor
nmxi = np.zeros((nxp,nyp))
nmxe = np.zeros((nxp,nyp))

# Given min-max of ion and electron velocities,
# create a grid in the velocity space to work with
# returns velocity axes, 3D meshgrid, as well as
# grid spacing
def create_vel_space(mm):
   dv={}; vel={}; dv3={}; vg={}
   for sp in ['i','e']:
      dv[sp] =[(mm[sp][i]-mm[sp][i-1])/nv[(i-1)//2] for i in range(1,6,2)]
      vel[sp]=[np.arange(nv[i])*dv[sp][i]+mm[sp][i*2]+dv[sp][i] for i in range(3)]
      vg[sp] = np.meshgrid(*vel[sp],indexing='ij')
      dv3[sp]=dv[sp][0]*dv[sp][1]*dv[sp][2]
   return vel,vg,dv3


# Helper functions to clean up the code below
def fv_from_hist(h,mm,sp):
   """ gets the 3D histogram of particles in velocity space 
       and returns normalized density, first moment, and thermal
       velocity """
   vel,vg,dv3 = create_vel_space(mm)
   f    = h/np.sum(h)/dv3[sp]
   vavg = [np.sum(f*vg[sp][i])*dv3[sp] for i in range(3)]
   vth  = np.mean([np.sqrt(np.sum(f*(vg[sp][i]-vavg[i])**2)*dv3[sp])\
             for i in range(3)])
   return f,vavg,vth

def printerr(string,x,y):
   print("An error has occurred while "+string+" at position {:.2f},{:.2f}".format(x,y))

def vels(a,sp): 
   """ Returns velocity columns of a named numpy
       array as a list"""
   return [a[sp][i] for i in ['vx','vy','vz']]

# Each rank opens its own h5 file named with the x0,y0 of the rank
O_F = h5py.File(outpath+outfilebase+"-{:05.2f},{:05.2f}.h5".format(x0_proc,y0_proc),"w")

# Save simulation parameters in the h5 file
params=O_F.create_group("params")
for k in list(rc.params.keys()):
   if type(rc.params[k]) in [str,dict,list]:
      pass
   else:
      params.create_dataset(k,data=rc.params[k])
params.create_dataset("ngrid", data=ngrid)
#print("Done saving parameters to output file")

if rank==0:
   print("# ngrid = {}".format(ngrid))
   print("# rank,\t x,\t y,\t nm_i,\t nm_e")
comm.Barrier()

# Create velocity space for computation
vel,vg,dv3 = create_vel_space(mm)
#   
for ix in xr:
   for iy in yr:
      x = ix*rc.dx; y = iy*rc.dy
      try:
         ixx = xr.index(ix); iyy=yr.index(iy)
         og=O_F.create_group("{:05.2f},{:05.2f}".format(x,y))
      # COMPUTE MIN,MAX,MEAN,MEDIAN OF FIELDS
         rc.min_max(x,fdx,y,fdy)
         for kk in rc.mmx.keys(): 
            mmgroup=og.create_group(kk)
            for var in rc.primitives:
               mmgroup.create_dataset(var,data=rc.mmx[kk][var])
      # READ PARTICLES
         dl = [x,y,0] 
         dh = [x+fdx,y+fdy,rc.lz]
         rv=rc.read_particles_fields(dump_num,dl,dh)
      # COMPUTE ION DIST FN 
         h=af.faf.hist3d(*vels(rc.rv,'i'),*nv,minmax=mm['i']) 
         og.create_dataset("fi",data=h)
         f,vavg,vth = fv_from_hist(h,mm,'i')
         nmxi[ixx,iyy] = NonMaxwellianity(f,vel['i'],vavg,vth)
      # COMPUTE ELECTRON DFN
         h=af.faf.hist3d(*vels(rc.rv,'e'),*nv,minmax=mm['e']) 
         og.create_dataset("fe",data=h)
         f,vavg,vth = fv_from_hist(h,mm,'e')
         nmxe[ixx,iyy] = NonMaxwellianity(f,vel['e'],vavg,vth)
         print("{:03d},\t {:05.2f},\t {:05.2f},\t {:05.4e},\t {:05.4e}"\
               .format(rank,x,y,nmxi[ixx,iyy],nmxe[ixx,iyy]))
      except Exception as e:
         # Expect above to fail for a few last points of pex,pey when
         # ngrid > sliden
         printerr("Computing things at",x,y)
         exc_type, exc_obj, exc_tb = sys.exc_info()
         fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
         print(exc_type, fname, exc_tb.tb_lineno) 
         print(str(e))
#    
O_F.close()

if rank > 0:
   comm.send([my_px,my_py,nmxi,nmxe],dest=0,tag=13)
else:
   nmi[:nxp,:nyp]=nmxi
   nme[:nxp,:nyp]=nmxe
   for src in range(1,comm.size):
      rcvdat=comm.recv(source=src,tag=13,status=status)
      px=rcvdat[0]; py=rcvdat[1]
      arri=rcvdat[2]; arre=rcvdat[3]
      nmi[px*nxp:(px+1)*nxp,py*nyp:(py+1)*nyp]=arri
      nme[px*nxp:(px+1)*nxp,py*nyp:(py+1)*nyp]=arre

if rank == 0:
   import h5py as h5
   F=h5.File("NM-ng={}.h5".format(ngrid),"w")
   F.create_dataset("nmi",data=nmi)
   F.create_dataset("nme",data=nme)
   F.create_dataset("xx",data=rc.xx[::sliden]+round(rc.dx*sliden/2.,5))
   F.create_dataset("yy",data=rc.yy[::sliden]+round(rc.dy*sliden/2.,5))
   F.close()
   print("All done! Closing the hdf5 file now!")
