import numpy as np

from unidades import ureg
from etapa1_malla.elementos import tipo_elemento
from etapa2_interpolacion.funciones_forma import NODOS_NATURALES
from etapa3_matrices.matriz_B import matriz_B
from etapa3_matrices.gen_matriz_dispersa import dofs_elemento


## Tensiones sigma = D B u_e evaluadas en los nodos de cada elemento (igual que en T3)
##   sigma = [sxx, syy, szz, txy, tyz, tzx] [Pa], B en las coordenadas naturales de cada nodo
## Un nodo compartido recibe un valor de cada elemento vecino -> se promedian
## xy: pint, U: vector global (3 n_nodos) [m], D: matriz_D [Pa]
## Devuelve sigma_nodos (n_nodos, 6) promediado y sigma_elem [(n_nodos_e, 6), ...] sin promediar
def calc_tensiones(xy, conectividad, U, D):
    xy_m = xy.to(ureg.meter).magnitude
    suma = np.zeros((len(xy_m), len(D)))
    cuenta = np.zeros(len(xy_m))
    sigma_elem = []
    for nodos in conectividad:
        tipo = tipo_elemento(nodos)
        xy_e = xy_m[list(nodos)]
        u_e = U[dofs_elemento(nodos)]
        s_e = np.array([D @ matriz_B(tipo, xy_e, *nat)[0] @ u_e for nat in NODOS_NATURALES[tipo]])
        sigma_elem.append(s_e)
        suma[list(nodos)] += s_e
        cuenta[list(nodos)] += 1
    return suma/cuenta[:, None], sigma_elem


## Von Mises [Pa] a partir de sigma (..., 6) = [sxx, syy, szz, txy, tyz, tzx]
## (o (..., 3) = [sxx, syy, txy] en tension plana, como en T3)
def von_mises(sigma):
    sigma = np.asarray(sigma)
    if sigma.shape[-1] == 3:
        sxx, syy, txy = sigma[..., 0], sigma[..., 1], sigma[..., 2]
        return np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*txy**2)
    sxx, syy, szz, txy, tyz, tzx = np.moveaxis(sigma, -1, 0)
    return np.sqrt(((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2)/2 + 3*(txy**2 + tyz**2 + tzx**2))
