import os, sys, fenics, numpy, scipy, pickle, time, torch
import matplotlib.pyplot as plt
import Mesh, TTDN
from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
N_processes = comm.Get_size()

sample_size = 1500
tol_list = [3.2e-2, 1.e-2, 3.2e-3, 1.e-3, 3.2e-4, 1.e-4, 3.2e-5, 1.e-5, 3.2e-6, 1.e-6]

mesh_data = Mesh.MeshData('porus', True)
TTDN.offline(mesh_data, sample_size, tol_list)
#RB.offline_test(mesh_data, sample_size, tol_list)
