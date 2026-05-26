"""This module contains a pure python implementation of the basic
cross-correlation algorithm for PIV image processing."""

# __licence_ = """
# Copyright (C) 2011  www.openpiv.net
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# """

import os
import numpy as np
import scipy
from scipy import io
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from PIL import Image
from scipy import special


class ContinuousFlowField:
    def __init__(self, data, inter=False):
        '''
        Checks if the continous flow should be created from a set of data points
        if so it interpolates them for a continuous flow field
        '''
        self.inter = inter
        if inter:
            self.f_uu = scipy.interpolate.interp2d(data[:,0],data[:,1],data[:,2])
            self.f_vv = scipy.interpolate.interp2d(data[:,0],data[:,1],data[:,3])

    '''
    Defining a synthetic flow field
    '''
    def f_uu(self, x, y):
        u = 2.5+0.5*np.sin((x**2+y**2)/0.01)
        return u

    def f_vv(self, x, y):
        v = 0.5+0.1*np.cos((x**2+y**2)/0.01)
        return v

    def get_uu_vv(self, x, y):
        # return the U and V velocity at a certain position
        if self.inter:
            return self.f_uu(x, y)[0],self.f_vv(x, y)[0]
        else:
            return self.f_uu(x, y),self.f_vv(x, y)

    def create_syn_quiver(self, number_of_grid_points, path=None):
        # return and save a synthetic flow map
        xx, yy = np.meshgrid(np.linspace(0, 1, number_of_grid_points), np.linspace(0, 1, number_of_grid_points))
        uu = np.zeros(xx.shape)
        vv = np.zeros(yy.shape)
        for r in range(xx.shape[0]):
            for c in range(xx.shape[1]):
                u, v = self.get_uu_vv(xx[r, c], yy[r, c])
                uu[r, c] = u
                vv[r, c] = v

        m = np.sqrt(np.power(uu, 2) + np.power(vv, 2))
        fig = pl.quiver(xx, yy, uu, vv, m, clim=[1.5, m.max()], scale=100, width=0.002, headwidth=6, minshaft=2)
        cb = pl.colorbar(fig)
        cb.set_clim(vmin=1.5, vmax=m.max())
    
        
        if not path:
            pl.savefig('syn_quiver.png', dpi=400)
            pl.close()
        else:
            pl.savefig(path + 'syn_quiver.png', dpi=400)
            pl.close()
        
        return xx,yy,uu,vv


def create_synimageparameters(input_data, x_bound, y_bound, image_size,
                              path='None', inter=False, den=0.008, per_loss_pairs=2,
                              par_diam_mean=8**(1.0/2), par_diam_std=0.5, par_int_std=0.25, dt=0.1):
    """Creates the synthetic image with the synthetic image parameters

    Parameters
    ----------
    input_data: None or numpy array
        If you have data from which to genrate the flow feild the synthetic image.
        It should be passed on as a numpy array with columns being (X grid position,Y grid position,U velocity at (X,Y) grid point,V velocity at (X,Y) grid point)
        Else, pass None and define a synthetic flow field in continuous_flow_field class.

    x_bound,y_bound: list/tuple of floats
        The boundries of interest in the synthetic flow field.

    image_size: list/tuple of ints
        The desired image size in pixels.
    
    path: str('None' for no generating data)
        Path to txt file of input data.

    inter: boolean
        False if no interpolation of input data is needed.
        True if there is data you want to interpolate from.   

    den: float
        Defines the number of particles per image.

    per_loss_pairs: float
        Percentage of synthetic pairs loss.

    par_diam_mean: float
        Mean particle diamter in pixels.

    par_diam_std: float
        Standard deviation of particles diamter in pixels.

    par_int_std: float
        Standard deviation of particles intensities.

    dt: float
        Synthetic time difference between both images.

    Returns
    -------
    ground_truth: continuous_flow_field class
        The synthetic ground truth as a continuous_flow_field class.

    cv:
        Convertion value to convert U,V from pixels/images to meters/seconds.

    x_1,y_1: numpy array
        Position of particles in the first synthetic image.

    U_par,V_par: numpy array
        Velocity speeds for each particle.

    par_diam1: numpy array
        Particle diamters for the first synthetic image.

    par_int1: numpy array
        Particle intensities for the first synthetic image.

    x_2,y_2: numpy array
        Position of particles in the second synthetic image.

    par_diam2: numpy array
        Particle diamters for the second synthetic image.

    par_int2: numpy array
        Particle intensities for the second synthetic image.
    """

    #Data processing
    
    if not path == 'None':
        f = open(path,'r')
        data = f.readlines()
        f.close()
        data = [line.split('\t') for line in data]
        data = np.array(data).astype(float)
        data = np.array([line for line in data.tolist() if 1.2*x_bound[1] >= line[1] >= 0.8*x_bound[0] and 1.2*y_bound[1] >= line[2] >= 0.8*y_bound[0]])
        
    else:
        data = input_data
        
    if inter:
        cff = ContinuousFlowField(data, inter=True)
    else:
        cff = ContinuousFlowField(None)
    
    #Creating syn particles
    
    num_of_par = int(image_size[0]*image_size[1]*den)
    num_of_lost_pairs = num_of_par*(per_loss_pairs/100)
    x_1 = np.random.uniform(x_bound[0]*0.8,x_bound[1]*1.2,num_of_par)
    y_1 = np.random.uniform(y_bound[0]*0.8,y_bound[1]*1.2,num_of_par)
    par_diam1 = np.random.normal(par_diam_mean,par_diam_std,num_of_par)
    particle_centers = np.random.uniform(size=num_of_par)-0.5
    par_int1 = np.exp(-particle_centers**2/(2*par_int_std**2))
    uu_par = np.zeros(x_1.shape)
    vv_par = np.zeros(y_1.shape)
    x_2 = np.zeros(x_1.shape)
    y_2 = np.zeros(y_1.shape)
    par_diam2 = np.zeros(par_diam1.shape)
    par_int2 = np.zeros(par_int1.shape)
    
    def move_par(i):
        uu_par[i], vv_par[i] = cff.get_uu_vv(x_1[i],y_1[i])
        x_2[i] = x_1[i]+uu_par[i]*dt
        y_2[i] = y_1[i]+vv_par[i]*dt
        par_diam2[i] = par_diam1[i]
        par_int2[i] = par_int1[i]
        
    cpl = 0
    for i in range(num_of_par):
        if cpl < num_of_lost_pairs:
            if -0.4 > particle_centers[i] or 0.4 < particle_centers[i]:
                per_to_lose = 1-(0.5 - np.abs(particle_centers[i]))/0.1
                if np.random.uniform()<min(per_loss_pairs/10,1)*per_to_lose:
                    x_2[i] = np.random.uniform(x_bound[0]*0.8,x_bound[1]*1.2)
                    y_2[i] = np.random.uniform(y_bound[0]*0.8,y_bound[1]*1.2)
                    par_diam2[i] = np.random.normal(par_diam_mean,par_diam_std)
                    par_int2[i] = np.exp(-(np.random.uniform()-0.5)**2/(2*par_int_std**2))
                    cpl += 1
                else:
                    move_par(i)
            else:
                move_par(i)
        else:
            move_par(i)
    
    print('Requested pair loss:', str(int(num_of_lost_pairs)), ' Actual pair loss:', str(cpl))
    xy_1 = np.transpose(np.vstack((x_1, y_1, uu_par, vv_par, par_diam1, par_int1)))
    xy_2 = np.transpose(np.vstack((x_2, y_2, par_diam2, par_int2)))
    
    #Choosing particles in boundary area
    
    bounded_xy_1 = np.asarray([xy for xy in xy_1 if x_bound[1]>=xy[0]>=x_bound[0] and y_bound[1]>=xy[1]>=y_bound[0]])
    bounded_xy_2 = np.asarray([xy for xy in xy_2 if x_bound[1]>=xy[0]>=x_bound[0] and y_bound[1]>=xy[1]>=y_bound[0]])
    
    #Tranforming coordinates into pixels
    
    x1 = ((bounded_xy_1[:,0]-x_bound[0])/(x_bound[1]-x_bound[0]))*image_size[0]
    y1 = ((bounded_xy_1[:,1]-y_bound[0])/(y_bound[1]-y_bound[0]))*image_size[1]

    x2 = ((bounded_xy_2[:,0]-x_bound[0])/(x_bound[1]-x_bound[0]))*image_size[0]
    y2 = ((bounded_xy_2[:,1]-y_bound[0])/(y_bound[1]-y_bound[0]))*image_size[1]
    
    conversion_value = min((x_bound[1]-x_bound[0])/image_size[0],(y_bound[1]-y_bound[0])/image_size[1])/dt
    
    return cff,conversion_value,x1,y1,bounded_xy_1[:,2],bounded_xy_1[:,3],bounded_xy_1[:,4],bounded_xy_1[:,5],x2,y2,bounded_xy_2[:,2],bounded_xy_2[:,3]



def generate_particle_image(height, width, xx, yy, particle_diameters, particle_max_intensities, bit_depth):
    """Creates the synthetic image with the synthetic image parameters
    Should be run with the parameters of each image (first,second) separately.

    Parameters
    ----------
    HEIGHT, WIDTH: int
        The number of pixels in the desired output image.

    X,Y: numpy array
        The X and Y positions of the particles, created by create_synimage_parameters().

    PARTICLE_DIAMETERS, PARTICLE_MAX_INTENSITIES: numpy array
		The intensities and diameters of the particles, created by create_synimage_parameters().
	
	BIT_DEPTH: int
		The bit depth of the desired output image.

    Returns
    -------
    Image: numpy array
        The desired synthetic image.
        :param width:

    """
    render_fraction = 0.75
    image_out = np.zeros([height, width])

    minrenderedcols = (xx - render_fraction * particle_diameters).astype(int)
    maxrenderedcols = (np.ceil(xx + render_fraction * particle_diameters)).astype(int)
    minrenderedrows = (yy - render_fraction * particle_diameters).astype(int)
    maxrenderedrows = (np.ceil(yy + render_fraction * particle_diameters)).astype(int)

    index_to_render = []

    for i in range(xx.size):
        if 1<minrenderedcols[i] and maxrenderedcols[i]< width and 1<minrenderedrows[i] and maxrenderedrows[i]< height:
            index_to_render.append(i)

    for i in range(len(index_to_render)):
        #print(i/len(index_to_render)*100)
        ind = index_to_render[i]
        max_int = particle_max_intensities[ind]
        par_diam = particle_diameters[ind]
        sqrt8 = np.sqrt(8)
        x = xx[ind]
        y = yy[ind]

        bl = max(minrenderedcols[ind],0)
        br = min(maxrenderedcols[ind],width)
        bu = max(minrenderedrows[ind],0)
        bd = min(maxrenderedrows[ind],height)

        for c in range(bl,br):
            for r in range(bu,bd):
                image_out[r,c] = image_out[r,c] + max_int * par_diam**2 * np.pi / 32 * \
                ( scipy.special.erf( sqrt8 * (c - x - 0.5) / par_diam ) - scipy.special.erf(sqrt8 * (c - x + 0.5) / par_diam)) * \
                ( scipy.special.erf( sqrt8 * (r - y - 0.5)/ par_diam) - scipy.special.erf(sqrt8 *(r - y + 0.5) / par_diam))

    noise_mean = 2**(bit_depth*0.3)
    noise_std = 0.25*noise_mean
    noise = noise_std * np.random.randn(height, width) + noise_mean
    return (image_out*(2**bit_depth * 2.8**2/8)+noise).astype(int)[::-1]