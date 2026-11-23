from functools import lru_cache

import numpy as np


## Coordenadas naturales (xi, eta) de los nodos locales, misma numeracion que elementos.py
## (antihorario, esquinas -> medios -> centro).
## Triangulos: xi = L2, eta = L3, L1 = 1 - xi - eta (coordenadas de area), nodos en (0,0) (1,0) (0,1)
## Cuadrilateros: cuadrado [-1, 1] x [-1, 1]
NODOS_NATURALES = {
    "T3": np.array([[0, 0], [1, 0], [0, 1]], dtype=float),
    "T6": np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]]),
    "Q4": np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float),
    "Q8": np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=float),
    "Q9": np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]],
                   dtype=float),
}

## Hexaedros 3D: cubo [-1, 1]^3 en (xi, eta, zeta). Misma idea que los cuadrilateros:
##   0-3: esquinas de la cara zeta = -1 (antihorario mirando desde +zeta), 4-7: las de zeta = +1
##   H20 agrega los medios de arista: 8-11 abajo, 12-15 arriba, 16-19 verticales
##   H27 agrega los centros de cara (20: zeta-, 21: eta-, 22: xi+, 23: eta+, 24: xi-, 25: zeta+)
##   y el centro del cubo (26)
_ESQ = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
_ARISTAS = [[0, -1, -1], [1, 0, -1], [0, 1, -1], [-1, 0, -1],
            [0, -1, 1], [1, 0, 1], [0, 1, 1], [-1, 0, 1],
            [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
_CENTROS = [[0, 0, -1], [0, -1, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 1], [0, 0, 0]]
NODOS_NATURALES["H8"] = np.array(_ESQ, dtype=float)
NODOS_NATURALES["H20"] = np.array(_ESQ + _ARISTAS, dtype=float)
NODOS_NATURALES["H27"] = np.array(_ESQ + _ARISTAS + _CENTROS, dtype=float)


## Lagrange cuadratico 1D con nodos en -1, 0, 1 evaluado en s: (l, dl/ds) para el nodo s_i
def _lagrange_1d(s_i, s):
    if s_i < 0:
        return s*(s - 1)/2, s - 0.5
    if s_i > 0:
        return s*(s + 1)/2, s + 0.5
    return 1 - s**2, -2*s


## Funciones de forma de los hexaedros en x = (xi, eta, zeta)
## Devuelve N (n_nodos,) y dN (3, n_nodos): filas dN/dxi, dN/deta, dN/dzeta
def _forma_3d(tipo, x):
    nat = NODOS_NATURALES[tipo]
    n = len(nat)
    N, dN = np.zeros(n), np.zeros((3, n))

    if tipo == "H8":                    # trilineal: producto de (1 + a xi)/2 en cada direccion
        f = 1 + nat*x                   # (8, 3)
        N = f.prod(axis=1)/8
        for d in range(3):
            otras = [k for k in range(3) if k != d]
            dN[d] = nat[:, d]*f[:, otras].prod(axis=1)/8
        return N, dN

    if tipo == "H20":                   # serendipity: esquinas (0-7) y medios de arista (8-19)
        # esquina (a, b, c):  N = (1 + a xi)(1 + b eta)(1 + c zeta)(a xi + b eta + c zeta - 2)/8
        c = nat[:8]
        f = 1 + c*x                     # (8, 3): 1 + a xi, 1 + b eta, 1 + c zeta
        g = c @ x - 2
        N[:8] = f.prod(axis=1)*g/8
        for d in range(3):
            otras = [m for m in range(3) if m != d]
            dN[d, :8] = c[:, d]*f[:, otras].prod(axis=1)*(g + f[:, d])/8
        # medio de arista: en la direccion donde la coordenada natural es 0 va (1 - xi^2) en vez de
        # (1 + a xi):  N = (1 - xi^2)(1 + b eta)(1 + c zeta)/4  (y lo mismo rotando ejes)
        c = nat[8:]
        f = np.where(c == 0, 1 - x**2, 1 + c*x)         # (12, 3) factor de cada direccion
        df = np.where(c == 0, -2*x, c)                   # su derivada
        N[8:] = f.prod(axis=1)/4
        for d in range(3):
            otras = [m for m in range(3) if m != d]
            dN[d, 8:] = df[:, d]*f[:, otras].prod(axis=1)/4
        return N, dN

    if tipo == "H27":                   # producto de Lagrange cuadraticos (como Q9, una dimension mas)
        for k, c in enumerate(nat):
            l, dl = zip(*[_lagrange_1d(c[d], x[d]) for d in range(3)])
            N[k] = l[0]*l[1]*l[2]
            dN[:, k] = [dl[0]*l[1]*l[2], l[0]*dl[1]*l[2], l[0]*l[1]*dl[2]]
        return N, dN

    raise ValueError(f"tipo '{tipo}' no existe")


## Funciones de forma en (xi, eta) para 2D o (xi, eta, zeta) para hexaedros
## Devuelve N (n_nodos,) y dN (dim, n_nodos): fila 0 = dN/dxi, fila 1 = dN/deta, [fila 2 = dN/dzeta]
## Se guardan en memoria (lru_cache): en los puntos de Gauss son las mismas para todos los elementos,
## asi que se calculan una sola vez. Por eso se devuelven de solo lectura.
@lru_cache(maxsize=4096)
def funciones_forma(tipo, xi, eta, zeta=None):
    N, dN = _funciones_forma(tipo, xi, eta, zeta)
    N, dN = np.array(N, dtype=float), np.array(dN, dtype=float)
    N.flags.writeable = dN.flags.writeable = False
    return N, dN


def _funciones_forma(tipo, xi, eta, zeta):
    if tipo.startswith("H"):
        return _forma_3d(tipo, np.array([xi, eta, zeta], dtype=float))
    if tipo in ("T3", "T6"):
        L = np.array([1 - xi - eta, xi, eta])
        dL = np.array([[-1, 1, 0],          # dL/dxi
                       [-1, 0, 1]])         # dL/deta
        if tipo == "T3":
            return L, dL.astype(float)
        L1, L2, L3 = L
        N = np.array([L1*(2*L1 - 1), L2*(2*L2 - 1), L3*(2*L3 - 1), 4*L1*L2, 4*L2*L3, 4*L3*L1])
        dN_dL = np.array([[4*L1 - 1, 0,        0],          # filas: nodo, columnas: d/dL1, d/dL2, d/dL3
                          [0,        4*L2 - 1, 0],
                          [0,        0,        4*L3 - 1],
                          [4*L2,     4*L1,     0],
                          [0,        4*L3,     4*L2],
                          [4*L3,     0,        4*L1]])
        return N, dL @ dN_dL.T

    nat = NODOS_NATURALES[tipo]
    xi_i, eta_i = nat[:, 0], nat[:, 1]
    if tipo == "Q4":
        N = (1 + xi_i*xi)*(1 + eta_i*eta)/4
        dN = np.array([xi_i*(1 + eta_i*eta)/4,
                       eta_i*(1 + xi_i*xi)/4])
        return N, dN

    if tipo == "Q8":
        N, dN = np.zeros(8), np.zeros((2, 8))
        for k, (a, b) in enumerate(nat):
            if a != 0 and b != 0:       # esquina
                N[k] = (1 + a*xi)*(1 + b*eta)*(a*xi + b*eta - 1)/4
                dN[:, k] = [a*(1 + b*eta)*(2*a*xi + b*eta)/4,
                            b*(1 + a*xi)*(a*xi + 2*b*eta)/4]
            elif a == 0:                # medio de lado horizontal (eta = b)
                N[k] = (1 - xi**2)*(1 + b*eta)/2
                dN[:, k] = [-xi*(1 + b*eta), b*(1 - xi**2)/2]
            else:                       # medio de lado vertical (xi = a)
                N[k] = (1 + a*xi)*(1 - eta**2)/2
                dN[:, k] = [a*(1 - eta**2)/2, -(1 + a*xi)*eta]
        return N, dN

    if tipo == "Q9":                    # producto de Lagrange cuadraticos en xi y eta
        N, dN = np.zeros(9), np.zeros((2, 9))
        for k, (a, b) in enumerate(nat):
            lx, dlx = _lagrange_1d(a, xi)
            ly, dly = _lagrange_1d(b, eta)
            N[k] = lx*ly
            dN[:, k] = [dlx*ly, lx*dly]
        return N, dN

    raise ValueError(f"tipo '{tipo}' no existe (opciones: {list(NODOS_NATURALES)})")
