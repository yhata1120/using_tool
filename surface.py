import numpy as np
import pandas as pd
import shutil
import os
import math
import glob
from scipy.spatial import distance
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import dot

class Surface:
    def __init__(self,LatP,bond_length,coordinates,cut_off):
        self.LatP = LatP
        self.bond_length = bond_length
        self.coordinates = coordinates
        self.cutoff_top = cut_off

        self.supercell = np.eye(3)
        self.supercell_atoms = np.eye(3)
        self.a = 0
        self.b = 0
        self.c = 0
        self.atoms_surface = np.eye(3)
        self.atoms_left = np.eye(3)
        self.atoms_right = np.eye(3)
        self.atoms_surface_m = np.eye(3)
        self.atoms_rbt = np.eye(3)
    
    def generate(self,dirname):
        tol = 1e-5
        self.supercell_atoms, self.a, self.b, self.c = read_atominfile(dirname)
        self.supercell_atoms = self.supercell_atoms[:,1:4]
        self.supercell_atoms = self.supercell_atoms[self.supercell_atoms[:,0]<self.a/2 + tol]
        
#         self.a = a
#         self.b = b
#         self.c = c

    def Write_to_lammps(self,supercell_atoms,filename):
        dim = np.array([1,1,1])
        X = supercell_atoms.copy()

        NumberAt = len(X) 

        dimx, dimy, dimz = dim

        xlo = 0.00000000
        xhi = self.a
        ylo = 0.00000000
        yhi = self.b 
        zlo = 0.00000000
        zhi = self.c

        yz = 0.0

        Type1 = np.ones(len(X), int).reshape(1, -1)

        Counter = np.arange(1, NumberAt + 1).reshape(1, -1)

        # data = np.concatenate((X_new, Y_new))
        W1 = np.concatenate((Type1.T, X), axis=1)
        FinalMat = np.concatenate((Counter.T, W1), axis=1)

        with open(filename, 'w') as f:
            f.write('#Header \n \n')
            f.write('{} atoms \n \n'.format(NumberAt))
            f.write('2 atom types \n \n')
            f.write('{0:.8f} {1:.8f} xlo xhi \n'.format(xlo, xhi))
            f.write('{0:.8f} {1:.8f} ylo yhi \n'.format(ylo, yhi))
            f.write('{0:.8f} {1:.8f} zlo zhi \n\n'.format(zlo, zhi))
            f.write('{0:.8f} {1:.8f} {2:.8f} xy xz yz \n\n'.format(0, 0, yz))            
            f.write('Atoms \n \n')
            np.savetxt(f, FinalMat, fmt='%i %i %.8f %.8f %.8f')
        f.close()
        
    def Expand_Super_cell(self,Atoms):
        # cell の大きさを入力
        basis = np.array([[self.a,0,0,],
                         [0,self.b,0,],
                         [0,0,self.c,]])
        """
        --- modified by Yaoshu ---
        expands the smallest CSL unitcell to the given dimensions.
        """

        # 拡大したい倍率
        x = np.arange(1) #等倍
        y = np.arange(-1,2,1) #-1～1で３倍
        z = np.arange(-1,2,1) #-1～1で３倍

        indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) * len(y) * len(z), 3)

        #1.get atoms
        Atoms = Atoms.repeat(len(indice),axis=0) + np.tile(np.dot(indice,basis),(len(Atoms),1))
        return Atoms
        
    def Extract_middle_atoms(self,expand_atoms):
        ylo = -2*self.LatP
        yhi = self.b + 2*self.LatP
        zlo = -2*self.LatP
        zhi = self.c + 2*self.LatP
        extracted_atoms = expand_atoms[((ylo) < expand_atoms[:,1]) & (expand_atoms[:,1]<(yhi)) & ((zlo) < expand_atoms[:,2]) & (expand_atoms[:,2]<(zhi))]
        return extracted_atoms
    
    def Extract_atoms(self,supercell_atoms):
        atoms_max = self.a
        # atoms はRBTごとに入力
        # atoms_middleはcnaを行う際の注目原子、extracted_atomsは相手原子
        atoms_cut = supercell_atoms[(supercell_atoms[:,0] > (atoms_max/2 - self.LatP*3)) & (supercell_atoms[:,0]< (self.LatP*3 + atoms_max/2))]
        expand_atoms = self.Expand_Super_cell(atoms_cut)
        extracted_atoms = self.Extract_middle_atoms(expand_atoms)
        
        return extracted_atoms
    
        
    def cna(self,atoms):
        atoms_max = self.a
        extracted_atoms = self.Extract_atoms(atoms)
        atoms_middle = atoms[(atoms[:,0] > (atoms_max/2 - self.LatP*2)) & (atoms[:,0]< (self.LatP*2 + atoms_max/2))]
#        write_middle(atoms_middle,'middle_region')
        # 距離行列を計算
        dist_M = distance.cdist(atoms_middle, extracted_atoms, metric='euclidean')
        # 0を除去
#        print(dist_M.size)
#        print(len(dist_M[dist_M!=0]),len(atoms_middle))
        dist_M = dist_M[dist_M!=0].reshape(dist_M.shape[0],dist_M.shape[1]-1)
        # 距離行列をもとにダングリングボンドを持つ原子を抽出
        cnaed_atoms = atoms_middle[np.where(np.count_nonzero(dist_M < self.bond_length*self.cutoff_top, axis=1)<self.coordinates)]
        
        return cnaed_atoms
    
    def make_mirror(self,atoms_surface):
        tol = 1e-5

        supercell = np.array([[self.a/2,0,0,],
                             [0,self.b,0,],
                             [0,0,self.c,],])
        atoms_fractional = dot(inv(supercell),atoms_surface.T).T
        atoms_mirror = atoms_fractional.copy()
        atoms_mirror = atoms_mirror[atoms_mirror[:,0]<1-tol]
        atoms_mirror[:,0] = 2-atoms_mirror[:,0]
        atoms_fractional = np.vstack((atoms_fractional,atoms_mirror))
        atoms_surface_m = dot(supercell,atoms_fractional.T).T

        return atoms_surface_m
    
    def get_surface(self):
        self.atoms_surface = self.cna(self.supercell_atoms)
        self.atoms_surface_m = self.make_mirror(self.atoms_surface)
        
        
    def get_RBT(self,dy = 0,dz = 0):
        tol = 1e-5
        atoms_surface_m = self.atoms_surface_m
        a = self.a
        b = self.b
        c = self.c
        
        self.atoms_surface_m = self.make_mirror(self.atoms_surface)
        

        atoms_left = atoms_surface_m[atoms_surface_m[:,0] < a/2 + tol]

        atoms_right = atoms_surface_m[atoms_surface_m[:,0] > a/2 + tol]

        atoms_right[:,1] = atoms_right[:,1] + dy
        atoms_right[:,2] = atoms_right[:,2] + dz
        atoms_rbt = np.vstack((atoms_left,atoms_right))
        self.atoms_rbt = atoms_rbt.copy()
        return atoms_rbt
    
    def show_picture_surface(self):
        atoms_rbt = self.atoms_rbt.copy()
        a = self.a
        b = self.b
        c = self.c
        atoms_left = atoms_rbt[atoms_rbt[:,0] < a/2 + 1e-5]
        atoms_right = atoms_rbt[atoms_rbt[:,0] > a/2 + 1e-5]

        y = atoms_left[:,1]
        z = atoms_left[:,2]
        x = -(a/2 - atoms_left[:,0])

        y_RBT = atoms_right[:,1]
        z_RBT = atoms_right[:,2]
        x_RBT = -(a/2 - atoms_right[:,0])

        fig, axes = plt.subplots(1, 2, tight_layout=True, gridspec_kw={ 'width_ratios': [6, 2]})
        #plt.figure(figsize = ((max(y) - min(y)),(max(z) - min(z))))

        axes[0] = axes[0]
        pcm = axes[0].scatter(y,z, c = abs(x), cmap = 'hot',s=100, vmin = 0, vmax = max(abs(x))+1)
        # fig.colorbar(pcm,cmap = 'hot')

        pcm = axes[0].scatter(y_RBT,z_RBT, c = abs(x_RBT), cmap = 'hot',s=50, vmin = 0,vmax = max(abs(x_RBT))+1)
        # fig.colorbar(pcm,cmap = 'hot')

        # plt.show()

        #plt.figure(figsize = (2*(max(x) - min(x)),(max(z) - min(z))))
        axes[1].grid()
        pcm = axes[1].scatter(x,z, c = abs(x), cmap = 'hot',s=100, vmin = 0, vmax = max(abs(x)) + 1)
        # fig.colorbar(pcm,cmap = 'hot')

        pcm = axes[1].scatter(x_RBT,z_RBT, c = abs(x_RBT), cmap = 'hot',s=50, vmin = 0, vmax = max(abs(x_RBT))+1)
        fig.colorbar(pcm)






LatP = 3.61999999500575
bond_length = LatP/np.sqrt(2)
coodinates = 12
cut_off = 1.05

surface = Surface(LatP,bond_length,coodinates,cut_off)

# generate class
surface.generate(os.getcwd())

# get surface atoms and mirrored surface atoms
surface.get_surface()

atoms_rbt = surface.get_RBT(dy = 0, dz = 0)

surface.show_picture_surface()
