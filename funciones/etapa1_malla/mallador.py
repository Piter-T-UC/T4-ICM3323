import numpy as np

from etapa1_malla.elementos import ELEMENTOS, caras_borde
from etapa2_interpolacion.funciones_forma import NODOS_NATURALES


## Mallador 3D estructurado. Toda la geometria va en floats SI (metros); el problema le pone
## las unidades despues (xy * ureg.meter). Misma idea que el mallador por bloques de T3:
## se arma una grilla de puntos X[i, j, k] (esquinas de los hexaedros), cada celda es un elemento,
## y los nodos repetidos (por ejemplo al cerrar un tubo) se funden con unir().


## Coordenadas de una direccion: pasa por todos los puntos de 'fijos' (bordes, uniones, pernos...)
## y parte cada tramo en partes iguales de largo <= h
def lineas(fijos, h):
    fijos = np.unique(np.round(np.asarray(fijos, dtype=float), 12))
    salida = [fijos[:1]]
    for x0, x1 in zip(fijos[:-1], fijos[1:]):
        n = max(1, int(np.ceil((x1 - x0)/h - 1e-9)))
        salida.append(np.linspace(x0, x1, n + 1)[1:])
    return np.concatenate(salida)


## Agrega los puntos medios de la grilla en las 3 direcciones (para H20 y H27):
## X (n0, n1, n2, 3) -> (2 n0 - 1, 2 n1 - 1, 2 n2 - 1, 3). Los medios son promedios, asi que
## las aristas quedan rectas y los nodos medios centrados
def _con_medios(X):
    for eje in range(3):
        X = np.moveaxis(X, eje, 0)
        Y = np.empty((2*len(X) - 1,) + X.shape[1:])
        Y[0::2] = X
        Y[1::2] = (X[:-1] + X[1:])/2
        X = np.moveaxis(Y, 0, eje)
    return X


## Hexaedros de una grilla estructurada
## X: (ni + 1, nj + 1, nk + 1, 3) esquinas de las celdas [m]; las direcciones i, j, k pasan a ser
## xi, eta, zeta del elemento, asi que deben formar un triedro derecho (si no det(J) < 0)
## Devuelve xy (n, 3), conectividad (n_elem, n_nodos) y celda (n_elem, 3) = indices (i, j, k)
## de cada elemento (sirve para saber en que zona de la grilla quedo cada uno)
def hexas_estructurados(X, tipo="H20"):
    if tipo not in ELEMENTOS:
        raise ValueError(f"tipo '{tipo}' no existe (opciones: {list(ELEMENTOS)})")
    p = 1 if tipo == "H8" else 2
    if p == 2:
        X = _con_medios(X)
    id_ = np.arange(np.prod(X.shape[:3])).reshape(X.shape[:3])
    off = ((NODOS_NATURALES[tipo] + 1)*p/2).astype(int)        # -1, 0, 1 -> 0, p/2, p

    conectividad, celda = [], []
    ni, nj, nk = [(n - 1)//p for n in X.shape[:3]]
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                conectividad.append([int(id_[p*i + a, p*j + b, p*k + c]) for a, b, c in off])
                celda.append((i, j, k))
    xy, conectividad = unir((X.reshape(-1, 3), conectividad))
    return xy, conectividad, np.array(celda)


## Caja con lineas de malla x, y, z (arreglos de coordenadas [m], por ejemplo de lineas() o linspace)
## distorsion: corre las lineas interiores en x, arriba +d*hx y abajo -d*hx alternando de signo, asi
## los elementos quedan como trapecios (prueba clasica de sensibilidad a la distorsion). 0 = regular,
## tiene que ser < 0.5 para que no se den vuelta
## Devuelve xy (n, 3) y conectividad
def caja(x, y, z, tipo="H20", distorsion=0.0):
    x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
    X = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
    if distorsion:
        hx = np.diff(x).mean()
        zeta = 2*(z - z[0])/(z[-1] - z[0]) - 1                  # -1 abajo, +1 arriba
        signo = (-1.0)**np.arange(len(x))
        signo[[0, -1]] = 0                                      # los bordes no se mueven
        X[..., 0] += distorsion*hx*signo[:, None, None]*zeta[None, None, :]
    xy, conectividad, _ = hexas_estructurados(X, tipo)
    return xy, conectividad


## Conectividad como matriz si todos los elementos tienen los mismos nodos (como en T3)
def _como_matriz(filas):
    filas = [[int(n) for n in nodos] for nodos in filas]
    if len({len(nodos) for nodos in filas}) == 1:
        return np.array(filas, dtype=int)
    return filas


## Junta varias mallas (xy, conectividad) fundiendo los nodos con la misma coordenada y deja solo
## los nodos usados por algun elemento. Igual al de T3 (funciona en 2D y 3D).
def unir(*mallas, tol=1e-9):
    xy = np.vstack([m[0] for m in mallas])
    offsets = np.cumsum([0] + [len(m[0]) for m in mallas])
    conectividad = [[n + off for n in nodos]
                    for (_, conec), off in zip(mallas, offsets) for nodos in conec]
    usados = sorted({n for nodos in conectividad for n in nodos})

    paso = tol*max(np.ptp(xy, axis=0).max(), 1e-300)
    claves = np.round(xy[usados]/paso).astype(np.int64)
    _, primero, inversa = np.unique(claves, axis=0, return_index=True, return_inverse=True)
    orden = np.argsort(primero)
    rango = np.empty_like(orden)
    rango[orden] = np.arange(len(orden))
    nuevo = dict(zip(usados, rango[inversa.ravel()]))
    xy_nuevo = xy[usados][primero[orden]]
    return xy_nuevo, _como_matriz([[nuevo[n] for n in nodos] for nodos in conectividad])


## ---------------------------------------------------------------------------------------------
## Seleccion por condicion geometrica en(x, y, z) -> bool, con x, y, z en metros.
## Ej: en = lambda x, y, z: np.isclose(z, 0)
## ---------------------------------------------------------------------------------------------

def _sin_unidades(xy):
    return np.asarray(xy.to("meter").magnitude if hasattr(xy, "magnitude") else xy, dtype=float)


## Nodos que cumplen en(x, y, z)
def nodos_en(xy, en):
    return [n for n, p in enumerate(_sin_unidades(xy)) if en(*p)]


## Caras de borde (de un solo elemento) con todos sus nodos cumpliendo en(x, y, z),
## como listas de nodos globales con la normal hacia afuera
def caras_en(xy, conectividad, en):
    xy_m = _sin_unidades(xy)
    return [cara for cara in caras_borde(conectividad) if all(en(*xy_m[n]) for n in cara)]


## Filas [nodo, rx, ry, rz] para el arreglo apoyos (1 = restringido)
def apoyos_en(xy, en, rx, ry, rz):
    return [[n, rx, ry, rz] for n in nodos_en(xy, en)]


## Entradas (nodos_cara, p) de presion en las caras de borde que cumplen en(x, y, z)
## p > 0 empuja hacia adentro del solido. Puede ser:
##   pint constante                       -> la misma en toda la cara
##   funcion p(x, y, z) -> pint (x, y, z en metros) -> evaluada en cada nodo de la cara
##                                           (presion variable, ej. hidrostatica)
def presion_en(xy, conectividad, en, p):
    caras = caras_en(xy, conectividad, en)
    if not callable(p):
        return [(cara, p) for cara in caras]
    xy_m = _sin_unidades(xy)
    salida = []
    for cara in caras:
        valores = [p(*xy_m[n]) for n in cara]
        unidad = valores[0].units
        salida.append((cara, np.array([v.to(unidad).magnitude for v in valores])*unidad))
    return salida
