import numpy as np

from unidades import ureg
from etapa1_malla.elementos import TIPO_CARA
from etapa2_interpolacion.funciones_forma import funciones_forma
from etapa3_matrices.puntos_gauss import puntos_gauss


## Fuerzas nodales equivalentes de una presion p sobre una cara (n_nodos_cara, 3) [N]
##   f_i = int_A N_i (-p n) dA
## La cara de un hexaedro es un Q4 / Q8 / Q9: se usan las mismas funciones de forma y puntos de
## Gauss que en 2D. Con x(xi, eta) sobre la cara:
##   n dA = (dx/dxi  x  dx/deta) dxi deta     (vector normal exterior por el area)
## asi que no hace falta normalizar: f_i = sum_g  w_g * N_i * (-p) * (dx/dxi x dx/deta)
## p > 0 empuja hacia adentro del elemento (compresion), los nodos de la cara van con su normal
## hacia afuera (como los entrega elementos.caras)
## p: un numero (presion constante) o un valor por nodo de la cara (presion variable, por ejemplo
## hidrostatica): en cada punto de Gauss se interpola con las mismas funciones de forma, p = N @ p_nodos
## (exacto si p es lineal en x, y, z, como la hidrostatica)
def f_eq_cara(xyz_cara, p):
    tipo = TIPO_CARA[len(xyz_cara)]
    f_cara = np.zeros((len(xyz_cara), 3))
    pts, pesos = puntos_gauss(tipo)
    for i in range(len(pesos)):
        N, dN = funciones_forma(tipo, *pts[i])
        p_i = N @ p if np.ndim(p) else p        # presion en el punto de Gauss
        T = dN @ xyz_cara                       # filas: dx/dxi, dx/deta
        nA = np.cross(T[0], T[1])               # normal exterior * |dA / dxi deta|
        f_cara += pesos[i] * np.outer(N, -p_i*nA)
    return f_cara


## Vector global (3 n_nodos) [N] de las presiones
## presiones = [(nodos_cara, p), ...] con p pint (MPa, kPa, kgf/cm^2, ...), un valor o uno por nodo
## de la cara, y nodos_cara ordenados con la normal hacia afuera (presion_en del mallador las arma asi)
def gen_f_presion(xy, presiones):
    xy_m = xy.to(ureg.meter).magnitude
    F = np.zeros(3*len(xy_m))
    for nodos_cara, p in presiones:
        f = f_eq_cara(xy_m[list(nodos_cara)], p.to(ureg.Pa).magnitude)
        for nodo, fn in zip(nodos_cara, f):
            F[3*nodo:3*nodo + 3] += fn
    return F
