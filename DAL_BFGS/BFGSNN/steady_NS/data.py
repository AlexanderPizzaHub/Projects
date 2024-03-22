import pickle as pkl
from scipy.stats import uniform
import numpy as np
import os
from utils import groundtruth as gt
import yaml

with open("../config.yml", "r") as stream:
    try:
        ymlfile = yaml.safe_load(stream)
        config = ymlfile['steady_NS']['datagen']
    except yaml.YAMLError as exc:
        print(exc)

print(config)
N = config['N']
dataname = config['dataname']

def sample_onepoint():
    x = uniform.rvs()
    y = uniform.rvs()
    return x,y

def domain_sampler(size):
    domain_data_x = list()
    domain_data_y = list()
    while len(domain_data_x)<size:
        x,y = sample_onepoint()
        domain_data_x.append(x)
        domain_data_y.append(y)

    domain_data = np.array([domain_data_x,domain_data_y]).T
    print(domain_data.shape)
    return domain_data

domain_data = domain_sampler(size=N)
print(domain_data.shape)


Nb = config['Nb']

def generate_random_bdry(Nb):
    '''
    Generate random boundary points.
    '''
    bdry_col = uniform.rvs(size=Nb*2).reshape([Nb,2])
    for i in range(Nb):
        randind = np.random.randint(0,2)
        if bdry_col[i,randind] <= 0.5:
            bdry_col[i,randind] = 0.0
        else:
            bdry_col[i,randind] = 1.0

    return bdry_col

bdry_col = generate_random_bdry(Nb)
print(bdry_col)

Ndiv = config['Ndiv']

div_col = domain_sampler(Ndiv)

if not os.path.exists('dataset/'):
    os.makedirs('dataset/')
with open('dataset/'+dataname,'wb') as pfile:
    pkl.dump(domain_data,pfile)
    pkl.dump(bdry_col,pfile)
    pkl.dump(div_col,pfile)
    pkl.dump(domain_data,pfile)

ugt,cgt,pgt,udat,fgt = gt.data_gen_interior(domain_data)
bdry_dat = gt.data_gen_bdry(bdry_col) #ALL boundary are zero

print(ugt.shape,cgt.shape,pgt.shape,udat.shape,fgt.shape)

with open("dataset/gt_on_{}".format(dataname),'wb') as pfile:
    pkl.dump(ugt,pfile)
    pkl.dump(cgt,pfile)
    pkl.dump(pgt,pfile)
    pkl.dump(udat,pfile)
    pkl.dump(fgt,pfile)
    pkl.dump(bdry_dat,pfile)

c_ygt,c_ugt,c_pgt,c_ydat,c_fgt = gt.data_gen_interior(domain_data)
with open("dataset/costdata_on_{}".format(dataname),'wb') as pfile:
    pkl.dump(c_ydat,pfile)