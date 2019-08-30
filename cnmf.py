import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import re
from numpy import genfromtxt
import cv2
from sklearn.decomposition import NMF,KernelPCA
from scipy.signal import savgol_filter
from pysptools import eea

class spectroscopic_map:
    
    def __init__(self,ir_file,folder):
        self.ir_file=ir_file
        self.folder=folder
        self.exclude=[]
        self.em_defined=False
        
    def match_histrogram(self,array_to_match,array):
        coeff=np.std(array)/np.std(array_to_match)
        array_to_match=array_to_match*coeff
        array_to_match=array_to_match-np.mean(array_to_match)+np.mean(array)
        return array_to_match    
    
    def crawldir(self,topdir=[], ext='sxm'):
        fn = dict()
        for root, dirs, files in os.walk(topdir):
                  for name in files:
                    if len(re.findall('\.'+ext,name)):
                        addname = os.path.join(root,name)
                        if root in fn.keys():
                            fn[root].append(addname)
                        else:
                            fn[root] = [addname]
        return fn   
    
    def load_data(self,hs_resolution_x=16,hs_resolution_y=16,
                  exclude=False,hi_res_spatial_x=256,hi_res_spatial_y=256,data_format='new',
                 limit_range=False,sg=False,limit_maps=False,orig_size_x=512,orig_size_y=512):
        self.hi_res_spatial_x=hi_res_spatial_x
        self.hi_res_spatial_y=hi_res_spatial_y
        if exclude!=False:
            self.exclude=exclude
        if data_format=='old':
            self.ir_data=np.genfromtxt(self.ir_file,skip_header=1,delimiter=',',encoding="utf-8")
            self.wavelength_list=np.array([self.ir_data[:,0]],dtype='int')
            self.wavelength_list[self.wavelength_list<0]=0
            self.hi_res_spectral_vector_length=self.wavelength_list[0].shape[0]
            B=self.ir_data[:,1:][:,0::4]
            B=B.T
        elif data_format=='new':
            self.ir_data=np.genfromtxt(self.ir_file,skip_header=1,delimiter=',',encoding="utf-8")
            self.wavelength_list=np.array([self.ir_data[:,0]],dtype='int')
            self.wavelength_list[self.wavelength_list<0]=0
            self.hi_res_spectral_vector_length=self.wavelength_list[0].shape[0]
            B=self.ir_data[:,1:].T
                       
        C=np.reshape(B[:hs_resolution_x*hs_resolution_y],[hs_resolution_x,hs_resolution_y,self.hi_res_spectral_vector_length])
        C[np.isnan(C)]=0
        C[C<0]=0
        
        if limit_range!=False:
            idx1=self.find_nearest(self.wavelength_list,limit_range[0])
            idx2=self.find_nearest(self.wavelength_list,limit_range[1])
            self.wavelength_list=[self.wavelength_list[0][idx1:idx2]]
            C=C[:,:,idx1:idx2]
            self.hi_res_spectral_vector_length=self.wavelength_list[0].shape[0]

        self.small_hs_map=C
        self.all_maps=[]
        self.hi_res_maps=[]

        files=self.crawldir(self.folder,'csv')
        count=0
        if limit_range!=False:
            for name in files[self.folder]:
                if 'am2.csv' in name:
                    if int(name[-12:-8])<=limit_range[0] or int(name[-12:-8])>=limit_range[1]:
                        self.exclude+=[int(name[-12:-8])]
        for name in files[self.folder]:
            if 'am2.csv' in name and int(name[-12:-8]) not in self.exclude:    
                count=count+1
                self.all_maps+=[name]
                try:
                    self.hi_res_maps+=[np.where(self.wavelength_list==int(int(name[-12:-8])))[1][0]]
                except IndexError:
                    self.hi_res_maps+=[self.find_nearest(self.wavelength_list,int(int(name[-12:-8])))]
  
        if sg==True:
            for i in range(self.small_hs_map.shape[0]):
                for j in range(self.small_hs_map.shape[1]):
                    self.small_hs_map[i,j]=self.sg(self.small_hs_map[i,j])

        if limit_maps==True:            
            self.hi_res_maps_array=np.zeros([orig_size_x,orig_size_y,count])
        else:
            self.hi_res_maps_array=np.zeros([hi_res_spatial_x,hi_res_spatial_y,count])

        count=0
        for name in self.all_maps:
            try:
                D=self.match_histrogram(genfromtxt(name, delimiter=','),self.small_hs_map[:,:,self.find_nearest(self.wavelength_list,int(int(name[-12:-8])))])
            except IndexError:
                D=self.match_histrogram(np.loadtxt(name, delimiter=','),self.small_hs_map[:,:,self.find_nearest(self.wavelength_list,int(int(name[-12:-8])))])
            self.hi_res_maps_array[:,:,count]=D
            count=count+1
            
        if limit_maps==True:
            self.hi_res_maps_array=self.hi_res_maps_array[:hi_res_spatial_x,:hi_res_spatial_y,:]  
            
    def sg(self,a):
            return savgol_filter(a,window_length=15, polyorder=3)
        
    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def do_nmf(self,use_defined_endmembers=True,**kwargs):
        
        N_NMF_components=kwargs.get('N_NMF_components', None)
        if use_defined_endmembers!=True:
            self.em_defined=False
        X_flat=np.reshape(self.small_hs_map,(self.small_hs_map.shape[0]*self.small_hs_map.shape[1],self.hi_res_spectral_vector_length))
        X_flat[X_flat<0]=0.00
        if self.em_defined==True:
            H_em=self.endmembers
            W_em=np.random.rand(len(X_flat),self.n_components)
            self.N_NMF_components=self.n_components
            model = NMF(n_components=self.n_components, init='custom', random_state=0,solver='mu')
            self.abundances_s = model.fit_transform(X_flat,H=H_em,W=W_em)
            self.components_s = model.components_
            N_NMF_components=self.n_components
        else:
            model = NMF(n_components=N_NMF_components, init='random', random_state=0,solver='mu')
            self.abundances_s = model.fit_transform(X_flat)
            self.components_s = model.components_
            self.n_components=N_NMF_components
 
        reshaped_abundances_s=np.reshape(self.abundances_s,(self.small_hs_map.shape[0],self.small_hs_map.shape[1],N_NMF_components))
        resized_abundances_s= cv2.resize(reshaped_abundances_s, (self.hi_res_spatial_x, self.hi_res_spatial_y))
        W_init=np.reshape(resized_abundances_s,(self.hi_res_spatial_x*self.hi_res_spatial_y,N_NMF_components))
        H_init=np.zeros([N_NMF_components,len(self.hi_res_maps)])
        
        for i in range(N_NMF_components):
            for j in range(len(self.hi_res_maps)):
                H_init[i,j]=self.components_s[i][self.hi_res_maps[j]]

        Y_flat=self.hi_res_maps_array
        Y_flat=np.reshape(Y_flat,(self.hi_res_spatial_x*self.hi_res_spatial_y,len(self.hi_res_maps)))
        Y_flat[Y_flat<0]=0.00

        model = NMF(n_components=N_NMF_components,init='custom',  random_state=0,solver='mu')
        self.abundances_map = model.fit_transform(Y_flat,H=H_init, W=W_init)
        self.components_map = model.components_

        restored_map=np.reshape(np.matmul(self.abundances_map,self.components_s),(self.hi_res_spatial_x,self.hi_res_spatial_y,self.hi_res_spectral_vector_length))
        self.restored_map=restored_map
        
    def plot_components(self):
        for i in range(len(self.components_s)):
            plt.plot(self.wavelength_list[0][1:],self.components_s[i][1:],label='Component '+str(i+1))
            plt.ylabel('Component intensity, V')
            plt.xlabel('Wavenumber, cm$^{-1}$')
            plt.legend()
        plt.show()

    def plot_maps(self,no_ticks=True):
        ZZ=self.abundances_map
        ZZ=np.reshape(ZZ,(self.hi_res_spatial_x,self.hi_res_spatial_y,self.n_components))

        for i in range(len(self.components_s)):
            plt.title('Component '+str(i+1))
            plt.imshow(ZZ[:,:,i],cmap='hot')
            if no_ticks:
                plt.yticks([])
                plt.xticks([])
            plt.colorbar()
            plt.tight_layout()
            plt.show()
            
    def define_endmembers(self,method='load',plotit=True,**kwargs):

        indices=kwargs.get('indices', None)
        components=kwargs.get('components', None)
        endmember_array=kwargs.get('endmember_array', None)
        
        self.em_defined=True
       
        if method=='nfindr':
            self.n_components=components
            nfindr = eea.NFINDR()
            extraction_map=np.copy(self.small_hs_map)
            extraction_map[extraction_map<0]=0.00
            self.endmembers = nfindr.extract(extraction_map, self.n_components, normalize=True)
        elif method=='pick':
            self.n_components=len(indices)
            self.endmembers=np.zeros([self.n_components,len(self.wavelength_list[0])])
            for i in range(len(indices)):
                self.endmembers[i,:]=self.small_hs_map[indices[i][0],indices[i][1],:]
        elif method=='load':
            self.n_components=len(endmember_array)
            self.endmembers=endmember_array
        if plotit:
            for i in range(len(self.endmembers)):
                plt.plot(self.wavelength_list[0][1:],self.endmembers[i,:][1:],label='Component '+str(i+1))
                plt.ylabel('Endmember intensity, V')
                plt.xlabel('Wavenumber, cm$^{-1}$')
                plt.legend()
            plt.show()