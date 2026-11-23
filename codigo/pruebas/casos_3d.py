import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "funciones"))

import time
from types import SimpleNamespace

import numpy as np

from unidades import ureg
from etapa1_malla.mallador import caja, lineas, nodos_en, presion_en
from etapa3_matrices.matriz_D import matriz_D
from etapa3_matrices.gen_matriz_dispersa import gen_matriz_dispersa
from etapa3_matrices.gen_f_presion import gen_f_presion
from etapa4_resolver.restricciones import Restriciones
from etapa4_resolver.sistema_disperso import resolver_disperso
from etapa4_resolver.tensiones import calc_tensiones


## Casos de prueba 3D: arman la malla, resuelven y devuelven los valores a comparar.
## Geometria en floats SI (m, Pa); E con unidades.


## K U = F para una malla con apoyos [nodo, rx, ry, rz] y presiones [(cara, p), ...]
## Devuelve U (n_nodos, 3) [m], sigma_nodos (n_nodos, 6) [Pa] y el tiempo [s]
def resolver(xy, conectividad, E, nu, apoyos, presiones):
    inicio = time.perf_counter()
    D = matriz_D(E, nu, "3D")
    K = gen_matriz_dispersa(xy, conectividad, D)
    F = gen_f_presion(xy, presiones)
    U = resolver_disperso(K, F, Restriciones(apoyos))
    sigma, _ = calc_tensiones(xy, conectividad, U, D)
    return U.reshape(-1, 3), sigma, time.perf_counter() - inicio


## Apoyos a partir de una lista de (condicion en(x, y, z), rx, ry, rz): un nodo que cumple varias
## condiciones junta todas sus restricciones (ej: simetria en x + perno en z)
def _apoyos(xy, condiciones):
    r = np.zeros((len(xy), 3), dtype=int)
    for en, *rest in condiciones:
        r[nodos_en(xy, en)] |= np.array(rest, dtype=int)
    fijos = np.flatnonzero(r.any(axis=1))
    return np.column_stack([fijos, r[fijos]])


## Pano de plancha entre pernos (grilla cuadrada de paso a), presion q en la cara z = t
## Por simetria se modela 1/4: de un perno (0, 0) al centro del pano (a/2, a/2), con un = 0 en los
## 4 bordes (son planos de simetria: no se corren ni giran en su normal)
## El perno esta en la cara z = t (como en el blindaje: perno y presion del lado del hormigon) y
## fija uz en el nodo (0, 0, t) (d_apoyo = 0) o en un parche de (d/2 x d/2) (1/4 del parche d x d)
## n: elementos aprox. por lado del cuarto de pano, n_t: elementos en el espesor
def resolver_pano(a, t, q, E, nu, n, n_t, tipo, d_apoyo=0.0):
    c = a/2
    fijos = [0, c] + ([d_apoyo/2] if d_apoyo > 0 else [])
    x = y = lineas(fijos, c/n)
    z = np.linspace(0, t, n_t + 1)
    xy_m, conectividad = caja(x, y, z, tipo)
    xy = (xy_m*ureg.m).to(ureg.mm)

    tol = 1e-9
    apoyos = _apoyos(xy, [
        (lambda x, y, z: np.isclose(x, 0) or np.isclose(x, c), 1, 0, 0),         # simetria en x
        (lambda x, y, z: np.isclose(y, 0) or np.isclose(y, c), 0, 1, 0),         # simetria en y
        (lambda x, y, z: np.isclose(z, t) and x <= d_apoyo/2 + tol and y <= d_apoyo/2 + tol, 0, 0, 1),
    ])
    presiones = presion_en(xy, conectividad, lambda x, y, z: np.isclose(z, t), q*ureg.Pa)
    U, sigma, tiempo = resolver(xy, conectividad, E, nu, apoyos, presiones)

    centro = nodos_en(xy, lambda x, y, z: np.isclose(x, c) and np.isclose(y, c))
    abajo = nodos_en(xy, lambda x, y, z: np.isclose(x, c) and np.isclose(y, c) and np.isclose(z, 0))[0]
    return SimpleNamespace(w=-U[centro, 2].mean(), sxx=sigma[abajo, 0], n_gdl=3*len(xy),
                           n_elem=len(conectividad), tiempo=tiempo)


## Voladizo empotrado en x = 0, presion q en la cara superior z = t
## nx, ny, n_t elementos en largo, ancho y espesor; distorsion: ver mallador.caja
def resolver_voladizo(L, b, t, q, E, nu, nx, ny, n_t, tipo, distorsion=0.0):
    x, y, z = np.linspace(0, L, nx + 1), np.linspace(0, b, ny + 1), np.linspace(0, t, n_t + 1)
    xy_m, conectividad = caja(x, y, z, tipo, distorsion)
    xy = (xy_m*ureg.m).to(ureg.mm)

    apoyos = _apoyos(xy, [(lambda x, y, z: np.isclose(x, 0), 1, 1, 1)])
    presiones = presion_en(xy, conectividad, lambda x, y, z: np.isclose(z, t), q*ureg.Pa)
    U, sigma, tiempo = resolver(xy, conectividad, E, nu, apoyos, presiones)

    punta = nodos_en(xy, lambda x, y, z: np.isclose(x, L))
    raiz = nodos_en(xy, lambda x, y, z: np.isclose(x, 0) and np.isclose(z, t))
    return SimpleNamespace(w=-U[punta, 2].mean(), sxx=sigma[raiz, 0].mean(), n_gdl=3*len(xy),
                           n_elem=len(conectividad), tiempo=tiempo)
