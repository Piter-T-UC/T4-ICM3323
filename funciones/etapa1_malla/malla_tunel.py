import numpy as np

from etapa1_malla.mallador import lineas, hexas_estructurados, _sin_unidades


## Malla de un blindaje (steel lining) de tunel rectangular hecho con planchas de acero.
## Todo en floats SI (metros).
##
## Ejes: x = ancho, y = alto, z = eje del tunel. El hueco interior es el rectangulo
## [-a/2, a/2] x [-b/2, b/2]; el acero va por fuera, de espesor t (lado del hormigon).
##
## El contorno interior se recorre con la coordenada s (antihorario mirando desde +z), partiendo en
## la esquina inferior derecha:
##   pared 0: derecha  (largo b)    pared 1: techo (largo a)
##   pared 2: izquierda (largo b)   pared 3: piso  (largo a)
## Cada pared es una plancha distinta en el contorno; en z el blindaje se arma con planchas de ancho
## 'ancho_plancha' (la ultima se queda con lo que sobre de L).
## Los pernos (stud bolts) van en la cara exterior; en cada plancha de cada pared se reparten con
## distancia 'borde' a los bordes y separacion <= 'paso' (paso_s en el contorno, paso_z en z).
##
## Coordenadas naturales de cada hexaedro: xi = s, eta = z, zeta = espesor (de adentro hacia afuera)

PAREDES = ["derecha", "techo", "izquierda", "piso"]


## Las 4 paredes del contorno interior: inicio, tangente, normal exterior y largo
def _paredes(a, b):
    inicio = np.array([[a/2, -b/2], [a/2, b/2], [-a/2, b/2], [-a/2, -b/2]])
    tangente = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]], dtype=float)
    normal = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    largo = np.array([b, a, b, a])
    return inicio, tangente, normal, largo


## Punto del contorno interior en s y direccion hacia afuera para el espesor
## En las esquinas la direccion es la diagonal (n1 + n2): asi la cara exterior queda en el
## rectangulo (a + 2t) x (b + 2t) y el elemento de la esquina es un trapecio
def _punto_contorno(s, a, b):
    inicio, tangente, normal, largo = _paredes(a, b)
    s0 = np.r_[0, np.cumsum(largo)]
    w = min(int(np.searchsorted(s0, s, side="right")) - 1, 3)
    u = s - s0[w]
    d = normal[w].copy()
    if np.isclose(u, 0):
        d += normal[w - 1]
    elif np.isclose(u, largo[w]):
        d += normal[(w + 1) % 4]
    return inicio[w] + u*tangente[w], d


##Funcion para colocar los pernos
def posiciones_pernos(largo, borde, paso):
    libre = largo - 2*borde
    if libre <= 0:
        return np.array([largo/2])
    n = max(1, int(np.ceil(libre/paso - 1e-9)))
    return np.linspace(borde, largo - borde, n + 1)


## Uniones de las planchas en z: 0, w, 2w, ..., L
def uniones_planchas(L, ancho_plancha):
    z = np.arange(0, L, ancho_plancha)
    return np.r_[z[~np.isclose(z, L)], L]


## Pernos (s, z) por defecto: grilla rectangular en cada plancha de cada pared
## Devuelve (n_pernos, 2)
def pernos_grilla(a, b, L, ancho_plancha, borde, paso_s, paso_z):
    largo = _paredes(a, b)[3]
    s0 = np.r_[0, np.cumsum(largo)]
    zu = uniones_planchas(L, ancho_plancha)
    pernos = []
    for w in range(4):
        for z0, z1 in zip(zu[:-1], zu[1:]):
            for s in s0[w] + posiciones_pernos(largo[w], borde, paso_s):
                for z in z0 + posiciones_pernos(z1 - z0, borde, paso_z):
                    pernos.append((s, z))
    return np.array(pernos)


## Malla del blindaje
##   a, b: ancho y alto interior, L: largo, t: espesor de plancha, ancho_plancha: ancho en z [m]
##   pernos: (n, 2) posiciones (s, z) de los pernos (por ejemplo pernos_grilla); cualquier otra
##           distribucion sirve (tresbolillo, mas densa en las esquinas, ...)
##   h: tamano maximo de elemento en s y z, n_t: elementos en el espesor
##   d_perno: si > 0 se agregan lineas a +-d/2 de cada perno, para apoyar un parche d x d en vez de
##            un solo nodo (un apoyo puntual en un solido da tension infinita en ese punto)
## Devuelve xy (n, 3), conectividad e info (dict con todo lo necesario para apoyos y graficos)
def malla_tunel(a, b, L, t, ancho_plancha, pernos, h, n_t=1, tipo="H20", d_perno=0.0):
    largo = _paredes(a, b)[3]
    s_esq = np.r_[0, np.cumsum(largo)]
    zu = uniones_planchas(L, ancho_plancha)
    pernos = np.asarray(pernos, dtype=float)

    # Lineas de la grilla: pasan por esquinas, uniones de planchas y pernos (+- d/2)
    desfases = [0.0] if d_perno <= 0 else [-d_perno/2, 0.0, d_perno/2]
    s_lin = lineas(np.r_[(s_esq, *[pernos[:, 0] + d for d in desfases])], h)
    z_lin = lineas(np.r_[(zu, *[pernos[:, 1] + d for d in desfases])], h)
    r_lin = np.linspace(0, t, n_t + 1)

    # Grilla de esquinas X[i, j, k] = P(s_i) + r_k d(s_i) en el plano, z_j en el eje
    X = np.zeros((len(s_lin), len(z_lin), len(r_lin), 3))
    for i, s in enumerate(s_lin):
        P, d = _punto_contorno(s, a, b)
        X[i, :, :, :2] = P + r_lin[None, :, None]*d
        X[i, :, :, 2] = z_lin[:, None]

    xy, conectividad, celda = hexas_estructurados(X, tipo)

    # Pared y plancha de cada elemento (segun el centro de su celda)
    s_c = (s_lin[celda[:, 0]] + s_lin[celda[:, 0] + 1])/2
    z_c = (z_lin[celda[:, 1]] + z_lin[celda[:, 1] + 1])/2
    pared = np.searchsorted(s_esq, s_c) - 1
    plancha = np.searchsorted(zu, z_c) - 1

    # Pernos en 3D (sobre la cara exterior)
    pernos_xyz = []
    for s, z in pernos:
        P, d = _punto_contorno(s, a, b)
        pernos_xyz.append([*(P + t*d), z])

    info = dict(a=a, b=b, L=L, t=t, tipo=tipo, ancho_plancha=ancho_plancha, d_perno=d_perno,
                s_lin=s_lin, z_lin=z_lin, s_esquinas=s_esq, z_uniones=zu,
                pernos_sz=pernos, pernos_xyz=np.array(pernos_xyz),
                pared=pared, plancha=plancha)
    return xy, conectividad, info


## Esta en la cara exterior (contacto con el hormigon)? x, y, z en metros
def en_exterior(info):
    a, b, t = info["a"], info["b"], info["t"]
    return lambda x, y, z: bool(np.isclose(abs(x), a/2 + t) or np.isclose(abs(y), b/2 + t))


## Nodos de la cara exterior tomados por cada perno: el nodo del perno o, si d_perno > 0, el
## parche d x d a su alrededor. Devuelve una lista de nodos por perno
def nodos_pernos(xy, info, tol=1e-6):
    xy_m = _sin_unidades(xy)
    ext = en_exterior(info)
    exteriores = np.array([n for n, p in enumerate(xy_m) if ext(*p)])
    r = info["d_perno"]/2 + tol
    salida = []
    for p in info["pernos_xyz"]:
        cerca = np.all(np.abs(xy_m[exteriores] - p) <= r, axis=1)
        salida.append(exteriores[cerca].tolist())
    return salida
