import numpy as np

from etapa2_interpolacion.funciones_forma import NODOS_NATURALES, funciones_forma


## Catalogo de elementos 3D (hexaedros). Cada cara de un hexaedro es un cuadrilatero de T3:
## H8 -> Q4, H20 -> Q8, H27 -> Q9 (mismas funciones de forma y Gauss para las presiones)
ELEMENTOS = {
    "H8":  dict(n_nodos=8,  esquinas=8, cara="Q4"),
    "H20": dict(n_nodos=20, esquinas=8, cara="Q8"),
    "H27": dict(n_nodos=27, esquinas=8, cara="Q9"),
}

## El largo de la fila de conectividad dice el tipo (igual que en T3)
TIPO_POR_NODOS = {info["n_nodos"]: tipo for tipo, info in ELEMENTOS.items()}
TIPO_CARA = {4: "Q4", 8: "Q8", 9: "Q9"}


def tipo_elemento(nodos):
    if len(nodos) not in TIPO_POR_NODOS:
        raise ValueError(f"elemento con {len(nodos)} nodos: no corresponde a ningun tipo (8 H8, 20 H20, 27 H27)")
    return TIPO_POR_NODOS[len(nodos)]


## Las 6 caras del cubo natural: (eje fijo, signo, eje u, eje v), con u x v = normal exterior.
## Mismo orden que los centros de cara del H27: zeta-, eta-, xi+, eta+, xi-, zeta+
_CARAS = [(2, -1, 1, 0), (1, -1, 0, 2), (0, 1, 1, 2), (1, 1, 2, 0), (0, -1, 2, 1), (2, 1, 0, 1)]


## Caras del elemento como listas de nodos locales, ordenadas como un Q4/Q8/Q9 de T3
## (antihorario mirando desde afuera -> la normal de la cara apunta hacia afuera del elemento)
def caras(tipo):
    nat3 = NODOS_NATURALES[tipo]
    nat2 = NODOS_NATURALES[ELEMENTOS[tipo]["cara"]]
    salida = []
    for fijo, signo, u, v in _CARAS:
        cara = []
        for cu, cv in nat2:
            p = np.zeros(3)
            p[fijo], p[u], p[v] = signo, cu, cv
            cara.append(int(np.flatnonzero(np.all(nat3 == p, axis=1))[0]))
        salida.append(cara)
    return salida


## Caras de borde (pertenecen a un solo elemento) como listas de nodos globales, normal hacia afuera
def caras_borde(conectividad):
    todas = [[int(nodos[k]) for k in cara] for nodos in conectividad
             for cara in caras(tipo_elemento(nodos))]
    cuenta = {}
    for cara in todas:
        clave = frozenset(cara)
        cuenta[clave] = cuenta.get(clave, 0) + 1
    return [cara for cara in todas if cuenta[frozenset(cara)] == 1]


## Revisa numero de nodos, indices validos y que det(J) > 0 en las esquinas y el centro de cada
## elemento (en 3D "antihorario" pasa a ser: (xi, eta, zeta) forman un triedro derecho)
def validar_malla(xy, conectividad):
    xy_m = np.asarray(getattr(xy, "magnitude", xy), dtype=float)
    n_nodos = len(xy_m)
    for e, nodos in enumerate(conectividad):
        try:
            tipo = tipo_elemento(nodos)
        except ValueError as err:
            raise ValueError(f"elemento {e}: {err}") from None
        if min(nodos) < 0 or max(nodos) >= n_nodos:
            raise ValueError(f"elemento {e} ({tipo}): nodo fuera de rango (hay {n_nodos} nodos)")
        if len(set(nodos)) != len(nodos):
            raise ValueError(f"elemento {e} ({tipo}): nodos repetidos en {list(nodos)}")
        xyz_e = xy_m[list(nodos)]
        for p in np.vstack([NODOS_NATURALES[tipo][:8], [[0, 0, 0]]]):
            detJ = np.linalg.det(funciones_forma(tipo, *p)[1] @ xyz_e)
            if detJ <= 0:
                raise ValueError(f"elemento {e} ({tipo}): det(J) = {detJ:.3g} <= 0 en {p}; "
                                 f"elemento invertido (revisar el orden de los nodos)")
