import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
from sklearn.metrics import mean_squared_error

import keras
from IPython.display import clear_output
import matplotlib as mpl

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

'''compare multiple labeled datasets
'x', 'y', 'labels' are of the same length
'labels' is integer labels, starting from 0,1,2... depending on number of classes
'class_names' length must be the same as number of classes
'col_names' length must be the same as the second dimension of 'x'
'f_names' length must be equal to 'y.shape[-1]'
'''
class Comparer:

    def __init__(self, x, y, labels, col_names, class_names, name):

        self.name = name
        self.n_samples = x.shape[0]
	
        self.x = x
        self.y = y
        self.labels = labels
	
        self.col_names = col_names                #column name for each of the properties
        self.class_names = class_names            #class name for each of the dataset
        self.colors = ['r','g','b','orange']      #add as needed
        self.linecolors = [(1, 0, 0, 0.8),(0, 0.5, 0, 0.8),(0, 0, 1, 0.8),(1, 0.5, 0, 0.8)]
        self.hatches = ['/','.','+','|']
        self.f_names = ["Oil", "Water", "Gas"]

    def plot_properties(self):
        fig = plt.figure(figsize=(12, 7))
        for i in range(len(self.col_names)):
            ax = plt.subplot(2, 3, i+1)    #adjust grid arrangement accordingly
            bb = np.linspace(np.min(self.x[:, i]), np.max(self.x[:, i]), 15)
            for j in range(len(self.class_names)):
                #print(self.col_names[i], self.class_names[j])
                ax.hist(self.x[self.labels==j, i].flatten(), alpha=0.4, bins=bb, hatch=None, 
                        color=self.colors[j], edgecolor=self.linecolors[j], histtype='stepfilled') 
            ax.legend(self.class_names)
            margin = (np.max(self.x[:, i]) - np.min(self.x[:, i]))*0.1      #set margin accordingly
            ax.set_xlim([np.min(self.x[:, i])-margin, np.max(self.x[:, i])+margin])
            #ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
            ax.set_ylabel("Frequency")
            ax.set_xlabel(self.col_names[i])
            ax.grid(False)
        fig.tight_layout() 
        fig.savefig('readme/'+self.name+'-hist.png')

    def plot_production(self):
        timesteps = np.linspace(0, self.y.shape[1]-1, self.y.shape[1])
        fig = plt.figure(figsize=(12, 3.5))
        for i in range(len(self.f_names)):
            ax = plt.subplot(1, 3, i+1)    #adjust grid arrangement accordingly
            for j in range(len(self.class_names)):
                #print(self.f_names[i], self.class_names[j])
                y_perclass = self.y[self.labels==j, :, i]
                for k in range(y_perclass.shape[0]):
                    ax.plot(timesteps, y_perclass[k, :], alpha=0.4, color=self.colors[j])
            #ax.legend(self.class_names)
            ax.set_xlim([0, self.y.shape[1]-1])
            ax.set_ylabel("Rate (bpd)")
            ax.set_xlabel("Timesteps")
            ax.set_title(self.f_names[i])	    
            ax.grid(False)
        fig.tight_layout() 
        fig.savefig('readme/'+self.name+'-prod.png')
	
    def plot_production_percentiles(self):
        timesteps = np.linspace(0, self.y.shape[1]-1, self.y.shape[1])
        fig = plt.figure(figsize=(12, 3.5))
        for i in range(len(self.f_names)):
            ax = plt.subplot(1, 3, i+1)    #adjust grid arrangement accordingly
            for j in range(len(self.class_names)):
                #print(self.f_names[i], self.class_names[j])
                y_perclass = self.y[self.labels==j, :, i]
                p10, p50, p90 = get_percentiles(y_perclass)
                ax.fill_between(timesteps, p10, p90, alpha=0.4, color=self.colors[j], label=self.class_names[j])
                ax.plot(timesteps, p10, alpha=0.8, color=self.colors[j])
                ax.plot(timesteps, p50, alpha=0.8, color=self.colors[j])
                ax.plot(timesteps, p90, alpha=0.8, color=self.colors[j])
            ax.legend()
            ax.set_xlim([0, self.y.shape[1]-1])
            ax.set_ylabel("Rate (bpd)")
            ax.set_xlabel("Timesteps")
            ax.set_title(self.f_names[i])
            ax.grid(False)
        fig.tight_layout() 
        fig.savefig('readme/'+self.name+'-prod-p10p90.png')

    def plot_pca_properties(self, ax):
        pca = PCA(2)
        pca.fit(self.x)
        v_pca = pca.transform(self.x)

        for j in range(len(self.class_names)):
            ax.scatter(v_pca[self.labels==j, 0], v_pca[self.labels==j, 1], s=30, c=self.colors[j], alpha=0.6)
        ax.legend(self.class_names)
        ax.set_xlabel('$z_1$')
        ax.set_ylabel('$z_2$')
        ax.grid(False)
        ax.set_title('PCA (properties)')

    def plot_pca_production(self, ax):
        y_f = np.reshape(self.y, [self.y.shape[0], -1])
        pca = PCA(2)
        pca.fit(y_f)
        v_pca = pca.transform(y_f)

        for j in range(len(self.class_names)):
            ax.scatter(v_pca[self.labels==j, 0], v_pca[self.labels==j, 1], s=30, c=self.colors[j], alpha=0.6)
        ax.legend(self.class_names)
        ax.set_xlabel('$z_1$')
        ax.set_ylabel('$z_2$')
        ax.grid(False)
        ax.set_title('PCA (prod. data)')

    def plot_contour_properties(self, ax):
        pca = PCA(2)
        pca.fit(self.x)
        v_pca = pca.transform(self.x)
	
        #create the meshgrid for the contour plot
        deltax = (np.max(v_pca[:, 0]) - np.min(v_pca[:, 0]))/10
        deltay = (np.max(v_pca[:, 1]) - np.min(v_pca[:, 1]))/10
        xmin = np.min(v_pca[:, 0]) - deltax
        xmax = np.max(v_pca[:, 0]) + deltax
        ymin = np.min(v_pca[:, 1]) - deltay
        ymax = np.max(v_pca[:, 1]) + deltay
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
	
        for j in range(len(self.class_names)):
            z1 = v_pca[self.labels==j, 0]
            z2 = v_pca[self.labels==j, 1]

            values = np.vstack([z1, z2])
            kernel = gaussian_kde(values, bw_method=0.3)    #adjust for contour intervals
            f = np.reshape(kernel(positions).T, xx.shape)
	
            ax.contour(xx, yy, f, colors=self.colors[j], alpha=0.6, levels=6)
    
        ax.set_xlabel('$z_1$')
        ax.set_ylabel('$z_2$')
        ax.grid(False)
        ax.set_title('PCA contour (properties)')
        
    def plot_contour_production(self, ax):
        y_f = np.reshape(self.y, [self.y.shape[0], -1])
        pca = PCA(2)
        pca.fit(y_f)
        v_pca = pca.transform(y_f)

        #create the meshgrid for the contour plot
        deltax = (np.max(v_pca[:, 0]) - np.min(v_pca[:, 0]))/10
        deltay = (np.max(v_pca[:, 1]) - np.min(v_pca[:, 1]))/10
        xmin = np.min(v_pca[:, 0]) - deltax
        xmax = np.max(v_pca[:, 0]) + deltax
        ymin = np.min(v_pca[:, 1]) - deltay
        ymax = np.max(v_pca[:, 1]) + deltay
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
	
        for j in range(len(self.class_names)):
            z1 = v_pca[self.labels==j, 0]
            z2 = v_pca[self.labels==j, 1]

            values = np.vstack([z1, z2])
            kernel = gaussian_kde(values, bw_method=0.3)    #adjust for contour intervals
            f = np.reshape(kernel(positions).T, xx.shape)
	
            ax.contour(xx, yy, f, colors=self.colors[j], alpha=0.6, levels=6)
    
        ax.set_xlabel('$z_1$')
        ax.set_ylabel('$z_2$')
        ax.grid(False)
        ax.set_title('PCA contour (prod. data)')

    def plot_pca(self):
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(2, 2, 1)
        self.plot_pca_properties(ax)
        ax = plt.subplot(2, 2, 2)
        self.plot_pca_production(ax)
        ax = plt.subplot(2, 2, 3)
        self.plot_contour_properties(ax)
        ax = plt.subplot(2, 2, 4)
        self.plot_contour_production(ax)
        fig.tight_layout() 
        fig.savefig('readme/'+self.name+'-pca.png')
	
#get percentiles
def get_percentiles(dt):
    p10 = np.percentile(dt, 10, axis=0)
    p50 = np.percentile(dt, 50, axis=0)
    p90 = np.percentile(dt, 90, axis=0)
    return p10, p50, p90




