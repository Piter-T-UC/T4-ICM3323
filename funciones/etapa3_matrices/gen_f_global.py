import numpy as np

from unidades import ureg


## Vector de fuerzas nodales (dim * n_nodos) en N, igual que en T3 pero con la dimension de xy
## f_nodales = [(nodo, Fx, Fy, Fz), ...] (pint, ejes globales)
def gen_f_global(xy, f_nodales):
    dim = xy.shape[1]
    F_global = np.zeros(dim*len(xy))
    for nodo, *cargas in f_nodales:
        for k, f in enumerate(cargas):
            F_global[dim*nodo + k] += f.to(ureg.newton).magnitude
    return F_global
