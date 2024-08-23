import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

#import and clean data
data10K = pd.read_table('20230912_CaF2_0.01Yb_10K.txt', delimiter='\s+', skiprows=14) #Import 10K datafile
data10K.columns = ['Wavelength', 'Intensity'] #name various columns for ease of selection
ndata10K = data10K[270:1300] #removes irrelevant data points, will likely need to be changed for higher temp



#x and y for ease of use
y = ndata10K['Intensity']
x = ndata10K['Wavelength']

#initial plot to help with guesses
plt.plot(x,y)


#Single Gaussian function
def gauss(x, mu, sigma, A):
    """Takes a set of paprameters and returns a single gaussian fit"""
    return A*np.exp(-(x-mu)**2/2/sigma**2)


#A combination of two gaussian functions 
#mu1 (the centre of the first gaussian distribution)
#mu2 (the centre of the second gaussian distribution)
#sigma1 (the variance of the 1st distribution)
#sigma1 (the variance of the 2nd distribution) 
#A1 (the amplitude of the first distribution)
#A2 (the amplitude of the second distribution)
def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    """Takes two sets of parameters, one for each potential gaussian and returns a combination of those gaussians"""
    return gauss(x,mu1,sigma1,A1) + gauss(x,mu2,sigma2,A2)



#Initial guess, this part is important as it will not converge unless the guess is very close
#The format of the guess is as follows: mu1, sigma1, A1, mu2, sigma2, A2
expected = (559, 46, 17983, 600, 250, 7500)

#uses the guess to calculate the actual paramaters and covariance matrix using scipy.optimize.curve_fit
#If the code only fits one gaussian, go to 'params' in variable explorer 
#Take the parameters from the gaussian that is fitted and plug them back into the initial guess
params, cov = curve_fit(bimodal, x, y, expected)
sigma=np.sqrt(np.diag(cov)) #Don't know if this is working correctly but it doesn't matter
x_fit = np.linspace(x.min(), x.max(), 500) #For plotting


#plot combined gaussian model
plt.plot(x_fit, bimodal(x_fit, *params), color='red', lw=2.5, label='Mixed Gaussian Model')

#individual Gaussian curves, might want to play around with lineweights/color to see data better but up to you
plt.plot(x_fit, gauss(x_fit, *params[:3]), color='red', lw=1, ls="--", label='Gaussian 1')
plt.plot(x_fit, gauss(x_fit, *params[3:]), color='red', lw=1, ls=":", label='Gaussian 2')

#Format the plot 
plt.title('Gaussian Mixed Model CaF2:Yb2+ @ 10K')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (counts)')
plt.legend()
plt.show() 