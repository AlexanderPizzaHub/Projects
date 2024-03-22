import pickle as pkl
from scipy.stats import uniform
import numpy as np
import os
from utils import groundtruth as gt
import yaml
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import bezier

with open("../config.yml", "r") as stream:
    try:
        ymlfile = yaml.safe_load(stream)
        config = ymlfile['NS']['datagen']
    except yaml.YAMLError as exc:
        print(exc)

N = config['N']
dataname = config['dataname']

#Starting from upper left corner, counterclockwise.
# upper left gamma_i
gamma_i = []
x = -1.0 
for i in range(101):
    y = 1.0 - (2)/100*i
    gamma_i.append([x,y])
gamma_i = np.array(gamma_i)

gamma_w1 = []
y = -1
for i in range(101):
    x = -1 + (2)/100*i
    gamma_w1.append([x,y])
gamma_w1 = np.array(gamma_w1)

gamma_o = [] 
x = 1
for i in range(201):
    y = -1 + (2)/200*(i)
    gamma_o.append([x,y])
gamma_o = np.array(gamma_o)

gamma_w2 = []
y = 1.0
for i in range(101):
    x = 1.0 - (1-0.5)/100*(i+1)
    gamma_w2.append([x,y])
gamma_w2 = np.array(gamma_w2)

gamma_f = [] 
b_vtc = np.asfortranarray(
    [
        [-0.5,0,0.5],
        [1,0.5,1],
    ]
)
curve = bezier.Curve(b_vtc,degree=2)


start_point = np.asfortranarray(
    [
        [-0.5],
        [1.0],
    ]
)
end_point = np.asfortranarray(
    [
        [0.5],
        [1.0],
    ]
)

s0 = curve.locate(start_point)
sT = curve.locate(end_point)

length = sT-s0
for i in range(101):
    gamma_f.append(curve.evaluate(sT - i/100*length).reshape(-1))
gamma_f = np.array(gamma_f)

gamma_w3 = [] 
y = 1
for i in range(101):
    x = -0.5 - (0.5)/100*(i)
    gamma_w3.append([x,y])
gamma_w3 = np.array(gamma_w3)

bd_col = np.concatenate([gamma_i,gamma_w1,gamma_o,gamma_w2,gamma_f,gamma_w3],axis=0)


'''
Domain sampling
'''
def sample_point(bbox):
    return np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])

'''def check_inside(point,outer_hp):
    if outer_hp.contains_point(point) == False:
        return False
    return True

def sample_domain(outer_hull,N):
    #outer hull is a hull objective covering the largest domain. 
    #removed_hulls is a list of hull object. 
    bbox = [outer_hull.min_bound,outer_hull.max_bound]
    outer_hp = Path(outer_hull.points[outer_hull.vertices])


    points = [] 
    for i in range(N):
        rand_point = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])
        while check_inside(rand_point,outer_hp) == False:
            rand_point = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])
        points.append(rand_point)

    points = np.array(points)
    return points '''

def check_inside(point,outer_hp,inner_hp):
    if outer_hp.contains_point(point) == False:
        return False
    for hp in inner_hp:
        if hp.contains_point(point) == True:
            return False
    return True

def sample_domain(outer_hull,removed_hulls,N):
    #outer hull is a hull objective covering the largest domain. 
    #removed_hulls is a list of hull object. 
    bbox = [outer_hull.min_bound,outer_hull.max_bound]
    outer_hp = Path(outer_hull.points[outer_hull.vertices])

    inner_hp = []
    for hull in removed_hulls:
        inner_hp.append(Path(hull.points[hull.vertices]))

    inner_hp = []
    for hull in removed_hulls:
        inner_hp.append(Path(hull.points[hull.vertices]))

    points = [] 
    for i in range(N):
        rand_point = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])
        while check_inside(rand_point,outer_hp,inner_hp) == False:
            rand_point = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])
        points.append(rand_point)

    points = np.array(points)
    return points 

outer_hull = ConvexHull(bd_col)



domain_data = sample_domain(outer_hull,[],N)

if not os.path.exists('dataset/'):
    os.makedirs('dataset/')
with open('dataset/'+dataname,'wb') as pfile:
    pkl.dump(domain_data,pfile)
    pkl.dump([gamma_i,gamma_w1,gamma_o,gamma_w2,gamma_f,gamma_w3],pfile) #outer boundaries
