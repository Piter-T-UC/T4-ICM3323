import numpy as np


## Simetria en el plano x = 0: se resuelve solo la mitad x >= 0 (con ux = 0 en el plano) y despues
## se reflejan los resultados para tener la estructura completa.
##   intensivos (u, sigma):  el valor del otro lado es el reflejado. Cambian de signo ux y las
##                           tensiones de corte con UN indice x (txy, tzx)
##   extensivos (F, R):      fuerzas nodales; en los nodos del plano cada mitad aporta lo suyo, asi
##                           que se suman las dos: la reaccion de simetria (Rx) se cancela sola

ESPEJO_U = np.array([-1.0, 1.0, 1.0])                       # (ux, uy, uz) o (Fx, Fy, Fz)
ESPEJO_SIGMA = np.array([1.0, 1.0, 1.0, -1.0, 1.0, -1.0])   # [sxx, syy, szz, txy, tyz, tzx]


## Malla completa a partir de la mitad x >= 0 (xy en floats). Los nodos del plano no se duplican.
## Devuelve xy, conectividad, espejo (nodo -> su reflejado, el mismo si esta en el plano),
## plano (bool por nodo) y otros (nodos fuera del plano, en el orden en que se agregan)
## Los elementos reflejados quedan con la numeracion al reves (det J < 0): sirven para graficar y
## ubicar valores, no para volver a calcular K
def reflejar_malla(xy, conectividad, tol=1e-9):
    xy = np.asarray(xy, dtype=float)
    plano = np.isclose(xy[:, 0], 0, atol=tol)
    otros = np.flatnonzero(~plano)
    espejo = np.arange(len(xy))
    espejo[otros] = len(xy) + np.arange(len(otros))
    conectividad = np.asarray(conectividad)
    return (np.vstack([xy, xy[otros]*ESPEJO_U]), np.vstack([conectividad, espejo[conectividad]]),
            espejo, plano, otros)


## Campo por nodo (n, k) intensivo: u (k = 3, signo ESPEJO_U) o sigma (k = 6, signo ESPEJO_SIGMA)
## En el plano se promedia con el reflejado, como haria el promedio nodal con los elementos de los
## dos lados: ux = 0 y txy = tzx = 0 en el plano
def reflejar_campo(v, plano, otros, signo):
    v = np.array(v, dtype=float)
    v[plano] = (v[plano] + v[plano]*signo)/2
    return np.vstack([v, v[otros]*signo])


## Fuerzas nodales (n, 3) extensivas (F, R): en el plano se suman las dos mitades
def reflejar_fuerza(f, plano, otros):
    f = np.array(f, dtype=float)
    f[plano] += f[plano]*ESPEJO_U
    return np.vstack([f, f[otros]*ESPEJO_U])
