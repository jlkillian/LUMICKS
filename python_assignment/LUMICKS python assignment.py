# LUMICKS python assignment.py
# Jessie Killian 01-08-2020

import numpy as np
from numpy import exp
import h5py
import png
from scipy.optimize import curve_fit


def openh5py(f):
# Returns the data group from an h5py data file
# Input: f = data file path
    file = h5py.File(f, 'r')
    return file['data']


def getpixelarray(data):
# Returns image pixels as 2d array
# Input: data = data group from h5py file
    
    sensor = data['sensor'] # extract image, mapping data
    mapping = data['mapping']
    
    height = data.attrs['image_height'] # extract image attributes
    width = data.attrs['image_width']
    
    inputarray = np.column_stack((mapping, sensor)) #join sensor and mapping arrays for easy indexing below
    
    summing = [] # this array stores raw sensor values that need to be accumulated for summing
    pixels = [] # this array is appended with the image pixel values once they are summed from the summing array
    
    for i in inputarray:
        if i[0] > 0: # pixels with a mapping value of 0 are ignored
            summing.append(i[1]) # each valid sensor value gets added to the summing array
            if i[0] == 2: # a sensor value of 2 indicates the pixel is complete
                pixels.append(np.sum(summing)) #sum the accumulated values and append to the pixel array 
                summing = [] # clear out the summing array to start the next pixel
                    
    return np.reshape(pixels, [height,width]) #return the pixel values as a 2d array


def savepng(pixels, f):
# Save a 2d array of pixel values as 16-bit png image
# Input: pixels = 2d pixel array, f = target save path
    png.from_array(pixels, 'L;16').save(f) 


def gaussian(x,a,x0,s,c):
# Defines the gaussian function for fitting.
# Input: x = x data array, a = amp, x0 = center, s = sigma, c = offset
    return a*exp(-(x-x0)**2/(2*s**2))+c


def fitgaus(xdata, ydata):
# Fits a 1d pixel value array with a gaussian. Returns the gaussian fit parameters
# Input: xdata = 1d array of pixel indices, ydata = 1d array of pixel values

    # Define some reasonable first guesses for the fit parameters
    cguess = min(ydata)
    aguess = max(ydata)-cguess
    x0guess = np.argmax(ydata)
    sguess = x0guess-next(x for x,y in enumerate(ydata) if y > aguess/2) # distance in pixels from image max to image half-max
    
    return curve_fit(gaussian, xdata, ydata, p0=[aguess,x0guess,sguess,cguess],bounds=(0, np.inf)) # parameters constrained to >=0


def getsigma(xdata, ydata):
# Returns sigma of gaussian fit
# Input: r = index of row to fit, pixels = 2d array of pixel values
    pars,cov = fitgaus(xdata, ydata)
    return pars[2]
