'''
This is a modified version of a LB python code for 2D flow around obstacles by
# 
# Copyright (C) 2013 FlowKit Ltd, Lausanne, Switzerland
# E-mail contact: contact@flowkit.com
# 
# https://palabos.unige.ch/get-started/lattice-boltzmann/lattice-boltzmann-sample-codes-various-other-programming-languages/
# 
that was modiried to generate flow fields around arrangements of discs for
the purposes of generating a flow field in a 2D model of a mask
This was done by Richard Sear in June and July 2020
email: r.sear@surrey.ac.uk and richardsear.me

It outputs a .npz file that is read in by naother code
which in turn uses the flow field to calculate lamnbda and penetrations

Below is the copyright text from the FlowKit code,
which also applies to this code:
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License, either
# version 3 of the License, or (at your option) any later version.
# 
'''


from numpy import *; from numpy.linalg import *
#import matplotlib.pyplot as plt; from matplotlib import cm
import time
import yaml 


def read_yaml(filename):
    with open(filename) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
        yamlfile= yaml.load(file, Loader=yaml.FullLoader)
    return yamlfile
##
filename= './params.yaml'
yaml_in=read_yaml(filename)
print(yaml_in)
print(yaml_in['d_f'])


maxIter =int(yaml_in['maxIter']) # 50000 # Total number of time iterations.
#maxIter=200


###### Flow definition #########################################################
print('code works by setting U0, disc diameter and LB tau (omega)')
d_f=float(yaml_in['d_f']) #15.0e-6
U0=float(yaml_in['U0'])#2.65e-2
#u_air=7.51e-2
nu_air=1.5e-5
#Re = 0.1 # Reynolds number.
Re = U0*d_f/(2.0*nu_air)
print('Re ',round(Re,5),' for air at ',round(U0*100.0,4),' cm/s')
print( ' and fibre diameter ',round(d_f*1.0e6,4),' um')
# lattice dimensions
q = 9 
#
#uLB     = 0.001
r=float(yaml_in['r']) #20
print('radius of discs in lattice units ',round(r,2))
# randomness in disc centres
random_disp=r*1.5
print('random displace of disc centres ',random_disp)
alpha=float(yaml_in['alpha'])   #0.1
print('area fraction of discs in lattice ',round(alpha,4))
a=sqrt((2.0*pi)/(sqrt(3.0)*alpha))*r
#
# check every sub
subIter=int(maxIter/25)
#
#nulb    = uLB*r/Re
#omega = 1.0 / (3.*nulb+0.5);
tau =float(yaml_in['tau']) # 0.75
#tau = 1.0
omega = 1.0/tau # relaxation rate
print('relaxation rate omega = 1/tau ',round(omega,5))
nulb=(2.0/omega-1.0)/6.0
print('kinematic viscosity nu in LB units ',round(nulb,5))
uLB = nulb*Re/r
#print('boundary velocity in LB units ',"{:e}".format(uLB))
print('boundary velocity in LB units ',uLB)
####
single_fibre=yaml_in['single_fibre']
if(single_fibre):
    n_obs=2
    print('number of discs ',n_obs)
    ny = int(round(2.0*a))
    nx = int(round(4.0*a))
    print('nx,  ny ',nx,ny)
else:

# lattice constant of hexagonal lattice
#a=2.01*r
    print('lattice constant ',round(a,3))
    area_frac=(pi*sqrt(3.0)/6.0)*(2.0*r/a)**2
    print('area fraction of discs in lattice ',round(area_frac,4))
# layers along y 
    n_ly=3
# number along x
    n_lx=5
    n_obs=n_lx*n_ly
    print('number of discs ',n_obs)
    ny = int(round(n_ly*a))
    nx = int(round(4.0*a)) + int(round(0.5*sqrt(3.0)*float(n_lx)*a))
    print('nx,  ny ',nx,ny)
###########################
print('run for time steps',maxIter)

###### Lattice Constants #######################################################
c = array([(x,y) for x in [0,-1,1] for y in [0,-1,1]]) # Lattice velocities.
for i in range(q):
    print('velocity vector ',i,' = ',c[i])
t = 1./36. * ones(q)                                   # Lattice weights.
t[asarray([norm(ci)<1.1 for ci in c])] = 1./9.; t[0] = 4./9.
print('weights ',t)
#
noslip = [c.tolist().index((-c[i]).tolist()) for i in range(q)] 
print('noslip onsite bounce back ',noslip)
i1 = arange(q)[asarray([ci[0]<0  for ci in c])] # Unknown on right wall.
print('lattice vectors pointing to - x i1',i1)
i2 = arange(q)[asarray([ci[0]==0 for ci in c])] # Vertical middle.
print('lattice vectors with 0 x comp   i2',i2)
i3 = arange(q)[asarray([ci[0]>0  for ci in c])] # Unknown on left wall.
print('lattice vectors pointing to + x i3',i3)
###### Function Definitions ####################################################
sumpop = lambda fin: sum(fin,axis=0) # Helper function for density computation.
def equilibrium(rho,u):              # Equilibrium distribution function.
    cu   = 3.0 * dot(c,u.transpose(1,0,2))
    usqr = 3./2.*(u[0]**2+u[1]**2)
    feq = zeros((q,nx,ny))
    for i in range(q): feq[i,:,:] = rho*t[i]*(1.+cu[i]+0.5*cu[i]**2-usqr)
    return feq
###



'''
PBC along y function
'''
def y_PBC(yPBC):
# PBCs along y
    if(yPBC<0): 
        yPBC=yPBC+ny
    elif(yPBC>ny):
        yPBC=yPBC-ny   
    return yPBC
"""
set up obstacles
"""
def obstacle_def(r,single_fibre):
# centres of discs
    cx=zeros(n_obs)
    cy=zeros(n_obs)
# to measure thickness
    xmax=0
    xmin=nx
# Coordinates of the cylinders
    if(single_fibre):
        print('running to find single (really 2!) fibre flow field')
# central one
        x0=nx/2
        y0=ny/2
        print('1st disc at ',round(x0,2),round(y0,2))
# ones above and below perp to flow direction
        cx[0]=x0#+random.uniform(-random_disp,random_disp)
#    print(random_disp,cx[i],random.uniform(-random_disp,random_disp))
        cy[0]=y0#+random.uniform(-random_disp,random_disp)
# flanking ones above and below perp to flow direction
        cx[1]=x0#+random.uniform(-random_disp,random_disp)
#    print(random_disp,cx[i],random.uniform(-random_disp,random_disp))
        cy[1]=1.0e-5#+random.uniform(-random_disp,random_disp)
    else:
        print('flow field for filter')
# ones above and below perp to flow direction
        i_obs=-1
        for j in range(0,n_ly):
            for i in range(0,n_lx):
                x0=2.0*a+float(i)*0.5*sqrt(3.0)*a
                i_obs=i_obs+1
                if( abs(i)%2 == 0):
                    cy[i_obs]=a*float(j)+random.uniform(-random_disp,random_disp)
                    cx[i_obs]=x0+random.uniform(-random_disp,random_disp)
                else:
                    cy[i_obs]=a*(float(j)+0.5)+random.uniform(-random_disp,random_disp)
                    cx[i_obs]=x0+random.uniform(-random_disp,random_disp)
                cy[i_obs]=y_PBC(cy[i_obs])
                print(i,j,cx[i_obs],cy[i_obs])
#####################################
##### now set True/False array for inside/outside disc
    inside_obs=zeros((nx,ny),dtype=bool)
    for i in range(0,nx):
        for j in range(0,ny):
            inside_obs[i,j]=False
            for i_obs in range(0,n_obs):
 # PBCs along y
                dj=j-cy[i_obs]
                if(dj>0.5*float(ny)):
                    dj=dj-float(ny)
                elif(dj<-0.5*float(ny)):
                    dj=dj+float(ny)
                if( (i-cx[i_obs])**2+dj**2 < r**2): 
                    inside_obs[i,j]=True
                    if(i>xmax): xmax=i
                    if(i<xmin): xmin=i
# determine thickness
    thickness=xmax-xmin
    return inside_obs,cx,cy,thickness
# now call it
print('generating obstacles')

obstacle,cx,cy,thickness=obstacle_def(r,single_fibre)
print('thickness of filter ',thickness)

# vel lambda function is used to both set LHS BC and to initialise vel in system


vel = fromfunction(lambda d,x,y: (1-d)*uLB,(2,nx,ny))
# original vel
#vel = fromfunction(lambda d,x,y: (1-d)*uLB*(1.0+1e-4*sin(y/ly*2*pi)),(2,nx,ny))

tstart = time.time()

# initial conditions are equilibrium for rho = 1 and u set at those of LHS BC using vel lambda
feq = equilibrium(1.0,vel)
#print(feq)
fin = feq.copy()
###### Main time loop ##########################################################
mean_ux=[]
time_list=[]
two_thirds=2.0/3.0
one_sixth=1.0//6.0
for tstep in range(maxIter):
# to check approach to steady state
    if(tstep > 1 and tstep%subIter == 0):
        check=u.copy()
# Right wall: outflow condition. Set LB f in last two row on RHS to be equal
# i1 are the 3 vectors with negative x components
    fin[i1,-1,:] = fin[i1,-2,:] 
# use lambda function sumpop to sum the 9 components of fin to get density rho at each point
    rho = sumpop(fin)         
# Calculate velocity.
    u = dot(c.transpose(), fin.transpose((1,0,2)))/rho
    '''
    Zou-He BC at left
    '''
# Left wall (ie. at x=0): impose velocity using lambda function
#    u[:,0,:] = vel[:,0,:]
    u[0,0,:] = uLB
    u[1,0,:] = 0.0
# compute density at left wall, ie for rho[0,:] overwrite values obtained by summing old fin above
    rho[0,:] = 1./(1.-u[0,0,:]) * (sumpop(fin[i2,0,:])+2.*sumpop(fin[i1,0,:]))
# Left wall: Zou/He boundary condition.
#    fin[i3,0,:] = fin[i1,0,:] + feq[i3,0,:] - fin[i1,0,:]
    fin[6,0,:] = fin[3,0,:] + two_thirds*rho[0,:]*uLB
    fin[7,0,:] = fin[5,0,:] + 0.5*(fin[2,0,:]-fin[1,0,:])+one_sixth*rho[0,:]*uLB
    fin[8,0,:] = fin[4,0,:] - 0.5*(fin[2,0,:]-fin[1,0,:])+one_sixth*rho[0,:]*uLB
    '''
    now compute equilibrium fs
    '''
    feq = equilibrium(rho,u) 
# Collision step
    fout = fin - omega * (fin - feq)
# BCs at obstacles: flip all vectors for all cells in obstacle
# eg f for [0 -1] becomes f for [0 1]
    for i in range(q): 
        fout[i,obstacle] = fin[noslip[i],obstacle]
# Streaming step.
    for i in range(q):
        fin[i,:,:] = roll(roll(fout[i,:,:],c[i,0],axis=0),c[i,1],axis=1)
# check convergence and write out stuff
    if(tstep > 1 and tstep%subIter == 0):
        mean_ux.append(mean(u[0,:,:])/uLB)
        time_list.append(time)
        print('time step ',tstep,'mean u_x/imposed u_x at LHS ',round(mean_ux[-1],6))
        max_diff=amax(abs(u-check))
        ind_diff = unravel_index(argmax(abs(u-check), axis=None), u.shape)
#        print('is element with largest change inside obstacle' ,obstacle[ind_diff[1],ind_diff[2]])
#        print('new u',u[:,ind_diff[1],ind_diff[2]],'old u',check[:,ind_diff[1],ind_diff[2]])
#        print('u ',u[1,10:40,50])
#        print('ch',check[1,10:40,50])
#    if (time%100==0): # Visualization
#        plt.clf(); plt.imshow(sqrt(u[0]**2+u[1]**2).transpose(),cmap=cm.Reds)
#        plt.savefig("vel."+str(time/100).zfill(4)+".png")
final_change_rate=(mean_ux[-1]-mean_ux[-2])/(subIter)
print('final rate of change of u_x/uLB ',"{:e}".format(final_change_rate))
tend = time.time()
print('run iterations in ',round((tend-tstart)/3600.0,3),' hours')
################
#print(u.shape)
#print(u[0,:,:].shape)
rplot=zeros((2,nx,ny))
uplot=u.copy()
plot_obs_x=[]
plot_obs_y=[]
for ix in range(0,nx):
    for iy in range(0,ny):
        rplot[0,ix,iy]=ix
        rplot[1,ix,iy]=iy
        if(obstacle[ix,iy]):
            uplot[0,ix,iy]=0.0
            uplot[1,ix,iy]=0.0
            plot_obs_x.append(ix)
            plot_obs_y.append(iy)

###
if(single_fibre):
    filename='flow_field_2rowRe'+str(round(Re,5))+'a'+str(round(alpha,5))+ \
    'df'+str(round(d_f*1.0e6,1))+'.npz'
else:
    filename='flow_field_filterRe'+str(round(Re,5))+'a'+str(round(alpha,5))+ \
    'df'+str(round(d_f*1.0e6,1))+'.npz'
#
print('\n','saving to ',filename)
#file = open(filename,'w')
savez_compressed(filename,Re=Re,disc_radius=r,lattice_const=a,
                 uBC=uLB,omegaLB=omega,nuLB=nulb,n_obs=n_obs,
                 disc_centres_x=cx,disc_centres_y=cy,
                 alpha=alpha,thickness=thickness,
                 random_disp=random_disp,U0=U0,d_f=d_f,tau=tau,
                 pos_vec=rplot,u_vec=uplot,x_obs=plot_obs_x,y_obs=plot_obs_y,allow_pickle=True)
#file.close()

