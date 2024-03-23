import pickle as pkl
from scipy.stats import uniform
import numpy as np
import os
from utils import groundtruth as gt
import yaml

with open("../config.yml", "r") as stream:
    try:
        ymlfile = yaml.safe_load(stream)
        config = ymlfile['linear_ctd']['datagen']
    except yaml.YAMLError as exc:
        print(exc)

N = config['N']
dataname = config['dataname']


def sample_onepoint():
    x = (6*uniform.rvs()-3)
    y = (6*uniform.rvs()-3)
    if x**2+y**2 >= 1.0 and x**2+y**2 <= 9.0: #radius in [1,3]
        return x,y
    else:
        return sample_onepoint()


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
    bdry_col = list()
    for i in range(Nb):
        is_interior = np.random.randint(0,4)
        if is_interior == 0: #inner circle
            r=1.0 
            theta = np.random.uniform(0,2*np.pi)
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            bdry_col.append([x,y])
        else:
            r = 3.0
            theta = np.random.uniform(0,2*np.pi)
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            bdry_col.append([x,y])
    bdry_col = np.array(bdry_col)
    return bdry_col

bdry_col = generate_random_bdry(Nb)
print(bdry_col)

if not os.path.exists('dataset/'):
    os.makedirs('dataset/')
with open('dataset/'+dataname,'wb') as pfile:
    pkl.dump(domain_data,pfile)
    pkl.dump(bdry_col,pfile)
    pkl.dump(domain_data,pfile)

ugt,cgt,pgt,udat,fgt = gt.data_gen_interior(domain_data)
bdry_dat = gt.data_gen_bdry(bdry_col)

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


