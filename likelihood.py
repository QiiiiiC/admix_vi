import numpy as np
import pandas as pd
import math
import topology
from simulation import all_ibd_segments, popn3_simple_data
import torch

#nodes.keys() is not ordered by nominal values, so map to some indeices.
def create_nodes_map(nodes):
    nodes_map = {}
    it = 0
    for i in nodes.keys():
        nodes_map[i] = it
        it += 1
    return nodes_map

# create a list of migration matrices, whenever there's a event(split/merge), create a migration matrix for that particular event.
# the starting element is always an identity matrix.
# each migration matrix is of dimension k*k where k is the total number of all populations(instead of starting populations).
def mig_matrix(nodes, events, nodes_map):
    n = len(nodes)
    events_num = int(len(events)-1-len([e for e in events if e[-1]==0])/2)
    out = [np.diag([1.0 for i in range(n)])for i in range(events_num + 1)]
    
    # cnt is for loop indices.
    cnt = 1
    for i in range(1,len(events)):
        # when there's a split backward in time, two elements are created in 'events', so skip one of them.
        if events[i][1] != events[i-1][1]:
            # this is a merge backward in time.
            if events[i][-1] == 1:
                dest = events[i][0]
                change = nodes[dest]['children']
                for j in change:
                    out[cnt][nodes_map[j]][nodes_map[j]]=0
                    out[cnt][nodes_map[j]][nodes_map[dest]]=1
                cnt += 1
            # this is a split backward in time
            else:
                change = nodes[events[i][0]]['children'][0]
                dest = nodes[change]['parents']
                out[cnt][nodes_map[change]][nodes_map[change]]=0
                for j in dest:
                    out[cnt][nodes_map[change]][nodes_map[j]]=nodes[j]['frac']
                cnt += 1
    return out

def frac_function(N,u,v,t):
    k1 = (2*N*v + 50)*50/((N*v + 50)**2) * (math.exp(-(N*v + 50)/(N*50)*t) - 1)
    k2 = -(2*N*u + 50)*50/((N*u + 50)**2) * (math.exp(-(N*u + 50)/(N*50)*t) - 1)
    k3 = v/(N*v + 50)*t*math.exp(-(N*v + 50)/(50*N)*t)
    k4 = -u/(N*u + 50)*t*math.exp(-(N*u + 50)/(50*N)*t)
    return k1 + k2 + k3 + k4

# A is the output of mig_matrix, N is the effective population sizes for all populations stored in vector.
# nodes_map map the population indices in nodes to arange(n).
# d is the starting number of populations, which is the ones we get at present time.
# T is the time difference sequence of events which should have length len(A), and the last element is always set to 1000000 referred to infinity.
# l is the length of chromosome we use in the unit of centiMorgans, u and v are range of IBD segments in the unit of centiMorgans.

def expected_ratio(N, d, T, nodes, events, u, v, l):
    A = mig_matrix(nodes, events,create_nodes_map(nodes))
    out = torch.zeros((d, d)).to(torch.float64)
    weight = torch.ones((d, d)).to(torch.float64)
    #the ith row of dist is the probability distribution of any individual initially from population i.
    dist = torch.eye(len(A[0])).to(torch.float64)
    

    for i in range(len(A)):
        dist = torch.matmul(dist, torch.tensor(A[i],dtype=torch.float64))
        for j in range(d):
            for k in range(j,d):
                #this is the probability distribution of popn i and popn j lie in one branch.
                prob = dist[j, :] * dist[k, :]
                
                #expected fractions updated
                fun = torch.tensor([frac_function(N[j], u, v, T[i]) for j in range(len(N))],dtype=torch.float64)
                out[j, k] += weight[j, k] * torch.matmul(prob, fun)
                out[k, j] = out[j, k]

                #weight updated
                ff = torch.tensor([(1 - math.exp(-T[i] / N[j])) for j in range(len(N))],dtype=torch.float64)
                gj = torch.matmul(prob, ff)
                weight[j, k] -= gj * weight[j, k]
                weight[k, j] = weight[j, k]
    return out

def expected_ratio_np(N, d, T, nodes, events, u, v, l):
    A = mig_matrix(nodes, events,create_nodes_map(nodes))
    out = [[0 for i in range(d)] for i in range(d)]
    weight = [[1 for i in range(d)] for i in range(d)]
    #the ith row of dist is the probability distribution of any individual initially from population i.
    dist = np.identity(len(A[0]))

    for i in range(len(A)):
        dist = np.matmul(dist,A[i])
        for j in range(d):
            for k in range(j,d):
                #this is the probability distribution of popn i and popn j lie in one branch.
                prob = np.multiply(dist[j,:],dist[k,:])
                
                #expected fractions updated
                fun = [frac_function(N[j],u,v,T[i]) for j in range(len(N))]
                out[j][k] += weight[j][k] * np.matmul(prob.T,fun)
                out[k][j] = out[j][k]

                #weight updated
                ff = [(1-math.exp(-T[i]/N[j])) for j in range(len(N))]
                gj = np.matmul(prob.T,ff)
                weight[j][k] -= gj*weight[j][k]
                weight[k][j] = weight[j][k]

    return out

N = [1000]*8
d = 3
T_original = [50,100,150,200]
T_diff = []
T = [50,50,50,50,1000000]
u = 10
v = 100
l = 100
nodes = topology.nodes_admix
events = topology.events_admix
expected_ratio(N,d,T,nodes,events,u,v,l)[1][2].item()




