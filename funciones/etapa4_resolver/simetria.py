import numpy as np


## Simetria en un plano (eje = valor, p. ej. x = 0 o z = L/2): se resuelve solo un lado (con u_eje = 0
## en el plano) y despues se reflejan los resultados para tener la estructura completa. Con dos
## planos (x = 0 y z = L/2) se resuelve un cuarto y se refleja una vez por plano.
##   intensivos (u, sigma):  el valor del otro lado es el reflejado. Cambian de signo u_eje y las
##                           tensiones de corte con UN indice eje (con eje = x: txy, tzx)
##   extensivos (F, R):      fuerzas nodales; en los nodos del plano cada mitad aporta lo suyo, asi
##                           que se suman las dos: la reaccion de simetria (R_eje) se cancela sola

ESPEJO_U = np.array([-1.0, 1.0, 1.0])                       # (ux, uy, uz) o (Fx, Fy, Fz), plano x = cte
ESPEJO_SIGMA = np.array([1.0, 1.0, 1.0, -1.0, 1.0, -1.0])   # [sxx, syy, szz, txy, tyz, tzx], plano x = cte


## Signos del espejo para un plano normal al eje (0 = x, 1 = y, 2 = z)
def espejo_u(eje):
    s = np.ones(3)
    s[eje] = -1.0
    return s


def espejo_sigma(eje):
    s = np.ones(6)
    for k, (i, j) in enumerate([(0, 1), (1, 2), (2, 0)]):   # txy, tyz, tzx: cambia si UNO de sus indices es eje
        if (i == eje) != (j == eje):
            s[3 + k] = -1.0
    return s


## Malla completa a partir de un lado del plano eje = valor (xy en floats). Los nodos del plano no se
## duplican. Devuelve xy, conectividad, espejo (nodo -> su reflejado, el mismo si esta en el plano),
## plano (bool por nodo) y otros (nodos fuera del plano, en el orden en que se agregan)
## Los elementos reflejados quedan con la numeracion al reves (det J < 0): sirven para graficar y
## ubicar valores, no para volver a calcular K
def reflejar_malla(xy, conectividad, eje=0, valor=0.0, tol=1e-9):
    xy = np.asarray(xy, dtype=float)
    plano = np.isclose(xy[:, eje], valor, atol=tol)
    otros = np.flatnonzero(~plano)
    espejo = np.arange(len(xy))
    espejo[otros] = len(xy) + np.arange(len(otros))
    conectividad = np.asarray(conectividad)
    xy_espejo = xy[otros].copy()
    xy_espejo[:, eje] = 2*valor - xy_espejo[:, eje]
    return (np.vstack([xy, xy_espejo]), np.vstack([conectividad, espejo[conectividad]]),
            espejo, plano, otros)


## Campo por nodo (n, k) intensivo: u (k = 3, signo espejo_u) o sigma (k = 6, signo espejo_sigma)
## En el plano se promedia con el reflejado, como haria el promedio nodal con los elementos de los
## dos lados: u_eje = 0 y las tensiones de corte con indice eje = 0 en el plano
def reflejar_campo(v, plano, otros, signo):
    v = np.array(v, dtype=float)
    v[plano] = (v[plano] + v[plano]*signo)/2
    return np.vstack([v, v[otros]*signo])


## Fuerzas nodales (n, 3) extensivas (F, R): en el plano se suman las dos mitades
def reflejar_fuerza(f, plano, otros, signo=ESPEJO_U):
    f = np.array(f, dtype=float)
    f[plano] += f[plano]*signo
    return np.vstack([f, f[otros]*signo])
