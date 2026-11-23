import numpy as np
from scipy.sparse import coo_matrix

from unidades import ureg
from etapa1_malla.elementos import tipo_elemento
from etapa3_matrices.matriz_B import matriz_B
from etapa3_matrices.puntos_gauss import puntos_gauss


## GDL globales de un elemento: [3*n0, 3*n0 + 1, 3*n0 + 2, 3*n1, ...] (ux, uy, uz por nodo)
def dofs_elemento(nodos, dim=3):
    return [dim*n + k for n in nodos for k in range(dim)]


## Matriz de rigidez del elemento (3n x 3n), igual que en T3 pero integrando en volumen
##   K_e = int_V B^T D B dV
## Se pasa al elemento de referencia (xi, eta, zeta) con dV = det(J) dxi deta dzeta y la integral se
## calcula con cuadratura de Gauss: suma sobre los puntos de Gauss de (peso * integrando)
##   K_e = sum_i  w_i * B_i^T D B_i * det(J_i)
## xy_e: (n, 3) coordenadas de los nodos [m], D: (6, 6) [Pa]
def k_elemento(xy_e, D):
    tipo = tipo_elemento(xy_e)
    n_gdl = 3*len(xy_e)
    K_e = np.zeros((n_gdl, n_gdl))

    # Puntos de Gauss del elemento de referencia: pts (m, 3) con [xi, eta, zeta] y pesos (m,)
    pts, pesos = puntos_gauss(tipo)

    for i in range(len(pesos)):
        # B y det(J) evaluados en el punto de Gauss i
        B, detJ, _ = matriz_B(tipo, xy_e, *pts[i])

        # Aporte del punto i a la integral: w * B^T D B det(J)
        K_e += pesos[i] * (B.T @ D @ B) * detJ

    return K_e


## Ensambla K_global (3 n_nodos x 3 n_nodos) dispersa (CSC) en floats SI (N/m)
## xy: (n_nodos, 3) pint, conectividad: una fila de nodos por elemento, D: matriz_D(..., "3D") [Pa]
## En 3D K siempre se guarda dispersa: con decenas de miles de nodos una matriz llena no cabe en memoria
def gen_matriz_dispersa(xy, conectividad, D):
    xy_m = xy.to(ureg.meter).magnitude
    filas, columnas, valores = [], [], []
    for nodos in conectividad:
        K_e = k_elemento(xy_m[list(nodos)], D)
        dofs = np.array(dofs_elemento(nodos))
        filas.append(np.repeat(dofs, len(dofs)))
        columnas.append(np.tile(dofs, len(dofs)))
        valores.append(K_e.ravel())
    n_lib = 3*len(xy_m)
    # coo_matrix suma las entradas repetidas: eso es el ensamble
    ij = (np.concatenate(filas), np.concatenate(columnas))
    return coo_matrix((np.concatenate(valores), ij), shape=(n_lib, n_lib)).tocsc()
