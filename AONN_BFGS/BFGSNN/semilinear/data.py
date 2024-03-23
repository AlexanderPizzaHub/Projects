import pickle as pkl
from scipy.stats import uniform
import numpy as np
import os
from utils import groundtruth as gt
import yaml

with open("../config.yml", "r") as stream:
    try:
        ymlfile = yaml.safe_load(stream)
        config = ymlfile['semilinear']['datagen']
    except yaml.YAMLError as exc:
        print(exc)

N = config['N']
dataname = config['dataname']

domain_data_x = uniform.rvs(size=N)
domain_data_y = uniform.rvs(size=N)

domain_data = np.array([domain_data_x,domain_data_y]).T
print(domain_data.shape)


Nb =config['Nb']
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