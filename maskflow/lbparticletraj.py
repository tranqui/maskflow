#!/usr/bin/env python3
'''
Impact on 2D fibre simulations
RPS June to August 2020 www.richardsear.me
email: r.sear@surrey.ac.uk

Loads in a flow field from Lattice Boltzmann code
then if:

1) if 2 obstacles in read in LB flow field calculates lambda

2) if > 2 obstacles in read in LB flow calculates penetration

in either case, reads in all parameters from .npz except
for diameter of particle

This program is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License, either
version 3 of the License, or (at your option) any later version.

'''


import numpy as np
import matplotlib.pyplot as plt#; from matplotlib import cm
import time
from math import floor
import os

'''
Read in flow field, parameters etc from .npz file from
LB simulations
'''

def slip_correction(diameter, temperature):
    """
    Slip correction for Stokes flow past a sphere (of Cunningham form).

    Args:
        diameter: particle diameter (m)
        A1, A2, A3: empirical correction parameters.
    Returns:
        Correction for slip effects in Stokes flow past a sphere.
    """
    A1=2.492
    A2=0.84
    A3=0.435
    boltzmann_constant = 1.38e-23 # J / K
    atmospheric_pressure = 101325 # Pa
    collisional_cross_section = np.pi * (3.64e-10)**2 # m^2 (using kinetic diameter of nitrogen, from wikipedia)
#    l = medium.mean_free_path(temperature)
    l=boltzmann_constant*temperature / \
    (np.sqrt(2) * collisional_cross_section * atmospheric_pressure)
    return 1 + l/diameter * (A1 + A2 * np.exp(-A3*diameter/l))



npzfiles=[]
for file in os.listdir("."):
    if file.endswith(".npz"):
        npzfiles.append(file)
print(npzfiles)
filename=npzfiles[0]
#filename='flow_field_2row.npz'
#filename='flow_field_filter.npz'
print('reading in ',filename)
# with hopefully close .npz file at end
with np.load(filename) as npzfile:
    print(npzfile.files)
    r=npzfile['disc_radius']
#    npzfile = np.load(f,allow_pickle=True)
    print('disc radius ',r)
    n_obs=npzfile['n_obs']
    a=npzfile['lattice_const']
    print(n_obs,' discs')
#print(npzfile['u_vec'])
    rplot=npzfile['pos_vec']
    xff=rplot[0,:,0]
    yff=rplot[1,0,:]
    nx=len(xff)
    ny=len(yff)
    print('lattice size ',nx,ny)
# us and vs are the x and y components of the flow field
    u_vec=npzfile['u_vec']
    uff=u_vec[0,:,:]
    vff=u_vec[1,:,:]

#
    plot_obs_x=npzfile['x_obs']
    plot_obs_y=npzfile['y_obs']
    cx=npzfile['disc_centres_x']
    cy=npzfile['disc_centres_y']
    alpha=npzfile['alpha']
    thickness=npzfile['thickness']
    Re=npzfile['Re']
    uBC=float(npzfile['uBC'])
    d_f=float(npzfile['d_f']*1.0e6)
    U0=float(npzfile['U0'])
    random_disp=float(npzfile['random_disp'])
# file now hopefully closed
#
#fibre diameter in micrometres
print('fibre diameter in um ',d_f)

# get particle diameter from terminal
d_p=float(input('enter particle diamter in um  '))
print(d_p,' particle diameter micrometres ',type(d_p))
ratio_pfibre=d_p/d_f
print('flow field speed U_0 ',np.round(U0,5),' m/s')
# slip correction
Cunningham_corr=slip_correction(d_p*1.0e-6,293)
print('Cunningham correction','{:9.3e}'.format(Cunningham_corr))
Stokes=6.2e6*Cunningham_corr*U0*d_p**2/d_f * 1.0e-6



# t0 is the relaxation time of particle to flow field
t0Stokes=Stokes*r/uBC
print('t0 Stokes ',round(t0Stokes,4))
print('')
print('parameter values:')
print('Stokes number particle, lengthscale is fibre radius ',round(Stokes,4))
print('Reynolds number ',Re)
#print('thickness/fibre d ',round(thickness/(2.0*r)))
#print('area fraction alpha ',"{:.4f}".format(alpha))
print('fibre     radius ',r)
r_coll=r*(1.0+ratio_pfibre)
print('collision radius ','{:9.3f}'.format(r_coll))
print('')
print('disc centres at x, y ',cx,cy)
dxLB=d_f/(2.0*r)
print('LB lattice constant in um ','{:9.3f}'.format(dxLB))
print('LB imposed speed          ','{:19.10f}'.format(uBC))
thickness=dxLB*(np.max(cx)-np.min(cx))
print(np.max(cx),np.min(cx))
print('filter thickness in um ','{:9.3f}'.format(thickness))
#
# set timestep
dt=min(100.0,0.025*t0Stokes)
dt2=0.5*dt
print('time step in LB units ','{:9.5f}'.format(dt))
# assess time step
# max fractional change in velocity is acceleration which is about
# difference in speed to ff, divided by t0 and divided by velocity
# so assume velocities cancel to get
max_frac_change_in_v=dt/t0Stokes
# max in position should be max speed times dt
max_frac_change_in_x=np.max(uff)*dt
print('estimated max fractional change in particle vel ','{:12.7f}'.format(max_frac_change_in_v))
print('estimated max fractional change in particle pos ','{:12.7f}'.format(max_frac_change_in_x))


print('alpha ',np.round(alpha,4))
#
# now count obstacles and either calculate penetration or lambda
if(n_obs> 2):
    root_find=False
    print('disorder in lattice        ',random_disp,' LB units')
    print('disorder in lattice/radius ',random_disp/r)
else:
    root_find=True
    print(' ')
    print(' root finding to find lambda')
    print(' ')
#



'''
PBCs for floating point y
'''
def y_PBC(yPBC):
# PBCs along y
    if(yPBC<0): 
        yPBC=yPBC+ny
    elif(yPBC>ny):
        yPBC=yPBC-ny   
    return yPBC
'''
Collision function for particle with all n_obs obstacles
'''
def particle_collide(x,y):
    collision=False
    rsq=r_coll**2
    for i_obs in range(0,n_obs):
        dxpf=abs(x-cx[i_obs])
# for speed
        if( dxpf < r_coll ):
            dypf=y-cy[i_obs]
# PBCs along y
            if(dypf>0.5*float(ny)):
                dypf=dypf-float(ny)
            elif(dypf<-0.5*float(ny)):
                dypf=dypf+float(ny)
# then check for collision
            if( dxpf**2+dypf**2 < rsq):
                collision=True       
    return collision
#
'''
Interpolate flow field read in from LB simulation
Needed as flow field on lattice, particle is not on lattice
'''
def u_interpolate(x,y):
    ix=int(x)
    iy=int(y)
# get 4 lattice points surrounding x,y
    if(x < ix):
        ixp=ix
        ixm=ix-1
    else:
        ixp=ix+1
        ixm=ix
# now for y, need :PBCS
    if(y < iy):
        iyp=iy
        if(iyp>ny-1): iyp=iyp-ny
        iym=iy-1
        if(iym<0): iym=iym+ny
    else:
        iyp=iy+1
        if(iyp>ny-1): iyp=iyp-ny
        iym=iy
        if(iym<0): iym=iym+ny
#
    delta_x=x-floor(x)
    delta_y=y-floor(y)
# bilinear interpolation
    usfp=uff[ixm,iym]*(1.0-delta_x)*(1.0-delta_y) \
        +uff[ixp,iym]*delta_x*(1.0-delta_y) \
        +uff[ixm,iyp]*(1.0-delta_x)*delta_y \
        +uff[ixp,iyp]*delta_x*delta_y
#
    vsfp=vff[ixm,iym]*(1.0-delta_x)*(1.0-delta_y) \
        +vff[ixp,iym]*delta_x*(1.0-delta_y) \
        +vff[ixm,iyp]*(1.0-delta_x)*delta_y \
        +vff[ixp,iyp]*delta_x*delta_y
#
    return usfp,vsfp

'''
Calculate acceleration of particle
'''
def accel_calc(x,y,vx,vy):
    usfp,vsfp=u_interpolate(x,y)
#
    ax=(usfp-vx)/t0Stokes
    ay=(vsfp-vy)/t0Stokes
    return ax,ay
'''
initial position for particle
initial velocity is local flow field velocity
'''
def initial_pos_vel(root_find,y):
# first position
# start at left hand side
    x=0.0
# start with flow field velocity at starting point
    vx,vy=u_interpolate(x,y)
    print(' ')
    print('starting position     ','{:9.3f}'.format(x),'{:9.3f}'.format(y), \
        ' vel ','{:9.5f}'.format(vx),'{:9.5f}'.format(vy))
#
    return x,y,vx,vy
'''
function for calculating a particle trajectory,
assuming it couples to given flow field at its
centre of mass 
'''
def part_traj(x,y,vx,vy):
# return trajectory but not at every step
    xtraj=[]
    ytraj=[]
    for it in range(1,1000000000):
# find acceleration
        ax,ay=accel_calc(x,y,vx,vy)
# now update x,y, vx, vy using midpoint aka modified Euler
# first mid points NB Wikipedia reckons velocity Verlet
# assumes acceleration depends on position only - not true here
# as it depends on velocity
        xmid=x+vx*dt2
        ymid=y+vy*dt2
        vxmid=vx+ax*dt2
        vymid=vy+ay*dt2
# PBCs along y
        ymid=y_PBC(ymid)
# now find acceleration at midpoint
        axmid,aymid=accel_calc(xmid,ymid,vxmid,vymid)
# now points at end of step
        x=x+vxmid*dt
        y=y+vymid*dt
# PBCs along y
        y=y_PBC(y)
        vx=vx+axmid*dt
        vy=vy+aymid*dt
# temp test
#        x=x+vx*dt
#        y=y+vy*dt
# PBCs along y
#        vx=vx+ax*dt
#        vy=vy+ay*dt
# end temp
#        if(it%100000 == 0):
#            print(it,x,y,vx,vy)
# update trajectory
        if(it%100 == 0):
            xtraj.append(x)
            ytraj.append(y)#
#            print(it,ix,iy,vx,vy,us[ix,iy],vs[ix,iy])
#            print(dx,dy,ax*dt2/vx,ay*dt2/vy)
# now check for collision
        ix=int(x)
# now check left box
        if(ix>nx-10):
            print('penetrated fibre      ','{:9.3f}'.format(x),'{:9.3f}'.format(y))
            penetrate=True
# put in dummy values for collision position, as we need to return values for it
            x_coll=0.0
            y_coll=0.0
# now quit loop
            break
# now check for collision
        if(particle_collide(x,y)):
            penetrate=False
            x_coll=x
            y_coll=y
            print('collided with fibre at','{:9.3f}'.format(x_coll), \
                    '{:9.3f}'.format(y_coll), \
                    ' vel ','{:9.5e}'.format(vx),'{:9.5e}'.format(vy))
            break
#############################
    return penetrate,x_coll,y_coll,xtraj,ytraj
###############
# number of trajectories depends on whether root finding
# or penetration calculations
if(root_find):
    n_traj=12
else:
    n_traj=100
#############
# to store locations of collisions
x_collide=[]
y_collide=[]
# to store trajectories in case we want to plot them
x_plot=[]
y_plot=[]
#
if(root_find):
    ymin=cy[0]
    ymax=ymin+ratio_pfibre*r*5
    ymax=ymin+r*1.8
    print(ymin,ymax)
    print('starting with ys','{:9.3f}'.format(ymin), \
          '{:9.3f}'.format(ymax), \
          ' which should bracket edge of coll zone')
else:
    print(' ')
    print(' study set of trajectories')
    print(' ')
    f_pene=0.0
#
start_time=time.time()
for i_traj in range(0,n_traj):
# starting position and velocity
    if(root_find):
        ytry=0.5*(ymin+ymax)
        x,y,vx,vy=initial_pos_vel(root_find,ytry)
    else:
        y=1.0e-10+float(ny)*float(i_traj)/float(n_traj)
        x,y,vx,vy=initial_pos_vel(root_find,y)
# now run single particle trajectory
    penetrated,x_coll,y_coll,xtraj,ytraj=part_traj(x,y,vx,vy)
# now check if particle has penetrated
    if(penetrated):
        if(root_find):
# update when root finding, edge of collision region inside current y
            ymax=ytry
        else:
# update fraction that penetrate filter
            f_pene=f_pene+1.0/float(n_traj)
    else:
# particle has collided
        if(root_find):
# update when root finding, edge of collision region outside current y
            ymin=ytry
# add new collision point
        x_collide.append(x_coll)
        y_collide.append(y_coll)
#
    x_plot.append(xtraj)
    y_plot.append(ytraj)
#####
# now print stuff out
# and write results t a .txt file
if(root_find):
    print('edge of collision region between ',ymin,' and ',ymax)
    width_collzone=2.0*(0.5*(ymax+ymin)-float(ny)/2.0)
    print('width coll zone                    ', \
          '{:9.3f}'.format(dxLB*width_collzone), ' um')
    print('width coll zone / particle radius  ', \
          '{:9.3f}'.format(width_collzone/(r*ratio_pfibre)))
    print('width coll zone / fibre diameter     ', \
          '{:9.3f}'.format(width_collzone/(2.0*r)))
    efficiency=width_collzone/(2.0*r)
    filename="lambda.txt"
    f=open(filename,"a+")
    string=str(round(Stokes,6))+' '+str(round(ratio_pfibre,5))+ \
    " "+str(round(d_f,3))+" "+str(round(d_p,3))+ \
    " "+str(round(dxLB*width_collzone,6))+" "+ \
    str(round(dxLB,5))+' '+str(np.round(alpha,6))+"\n"
    f.write(string)
    f.close()
else:
    efficiency=1.0-f_pene
    print('filter efficiency ',round(efficiency,4))
    filename="filter_eff.txt"
    f=open(filename,"a+")
    string=str(round(Stokes,6))+' '+str(round(ratio_pfibre,5))+ \
    " "+str(round(d_f,3))+" "+str(round(d_p,3))+ \
    " "+str(round(f_pene,6))+" "+ \
    str(np.round(dxLB,5))+" "+str(np.round(alpha,6))+ \
    " "+str(np.round(thickness,6))+"\n"
    f.write(string)
    f.close()
############
end_time=time.time()
run_time=end_time-start_time
print('run time ',round(run_time/3600.0,4),' hours')
print('particle radius/fibre radius ',round(ratio_pfibre,5))
#code below will plot LB flow field plus particle trajectories
# if true_plot=True, otherwise it won't
##############################
true_plot=True
true_plot=False
if(true_plot):
    plt.figure(figsize=(10*ny/nx,10))#,dpi=250)
    plt.streamplot(yff,xff,vff,uff,density=2.5)
#plt.plot(ytraj,xtraj,lw=4)
    for i in range(0,n_traj):
        plt.plot(y_plot[i][:],x_plot[i][:],lw=4)
#        print(x_plot[i][-1])
    for i in range(0,len(x_collide)):
        plt.scatter(y_collide,x_collide,s=300*ratio_pfibre**2,color='yellow',zorder=10)
    plt.scatter(plot_obs_y,plot_obs_x,s=2,color='brown')
    plt.xticks(())
#plt.xlim([1,300])
    plt.yticks(())
    plt.tight_layout()
#    plt.savefig('stream.png')
    plt.show()
