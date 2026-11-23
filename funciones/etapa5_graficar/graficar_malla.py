import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from etapa1_malla.elementos import caras, tipo_elemento


## Caras de borde con el elemento al que pertenecen: [(e, [nodos de la cara]), ...]
def _caras_borde_con_elemento(conectividad):
    todas = [(e, [int(nodos[k]) for k in cara]) for e, nodos in enumerate(conectividad)
             for cara in caras(tipo_elemento(nodos))]
    cuenta = {}
    for _, cara in todas:
        cuenta[frozenset(cara)] = cuenta.get(frozenset(cara), 0) + 1
    return [(e, cara) for e, cara in todas if cuenta[frozenset(cara)] == 1]


## Malla 3D: superficie exterior de los elementos, apoyos y presiones
## xy: (n, 3) pint. Se dibuja con el eje y del modelo hacia arriba (el z de matplotlib es el y del modelo)
## apoyos: arreglo [nodo, rx, ry, rz] -> puntos rojos
## presiones: [(nodos_cara, p), ...] -> flechas en el centro de las caras (como mucho max_flechas)
## color_elem: un numero por elemento (por ejemplo la plancha a la que pertenece) para colorear
def graficar_malla(xy, conectividad, apoyos=None, presiones=None, color_elem=None,
                   titulo="Malla 3D", vista=(22, -35), max_flechas=120):
    unidad = f"{xy.units:~}"
    xyz = np.asarray(xy.magnitude, dtype=float)
    P = lambda pts: pts[..., [0, 2, 1]]                   # (x, y, z) modelo -> (x, z, y) grafico

    fig = plt.figure(figsize=(12, 9), layout="constrained")
    ax = fig.add_subplot(projection="3d")

    caras_b = _caras_borde_con_elemento(conectividad)
    poligonos = [P(xyz[cara[:4]]) for _, cara in caras_b]       # las 4 esquinas de cada cara
    if color_elem is None:
        colores = "#cfd8e3"
    else:
        valores = np.asarray(color_elem, dtype=float)[[e for e, _ in caras_b]]
        cmap = plt.get_cmap("tab10" if len(np.unique(color_elem)) <= 10 else "viridis")
        colores = cmap((valores - valores.min())/max(np.ptp(valores), 1e-30)*0.999)
    ax.add_collection3d(Poly3DCollection(poligonos, facecolors=colores, edgecolors="0.35",
                                         linewidths=0.15, alpha=0.9))

    if apoyos is not None and len(apoyos):
        pts = P(xyz[np.asarray(apoyos)[:, 0]])
        ax.scatter(*pts.T, s=1.5, c="#c01c28", depthshade=False, label=f"Apoyos ({len(apoyos)} nodos)")

    if presiones:
        paso = max(1, len(presiones)//max_flechas)
        centros, normales = [], []
        for cara, _ in presiones[::paso]:
            pts = xyz[cara[:4]]
            n = np.cross(pts[1] - pts[0], pts[3] - pts[0])
            centros.append(pts.mean(axis=0))
            normales.append(n/np.linalg.norm(n))
        centros, normales = np.array(centros), np.array(normales)
        largo = 0.08*np.ptp(xyz, axis=0).max()
        inicio = centros + largo*normales                 # la flecha llega a la cara desde afuera
        ax.quiver(*P(inicio).T, *P(-largo*normales).T, color="#1a5fb4", lw=0.6,
                  arrow_length_ratio=0.25, label=f"Presion ({len(presiones)} caras)")

    lim = np.c_[xyz.min(axis=0), xyz.max(axis=0)]
    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[2])
    ax.set_zlim(*lim[1])
    ax.set_box_aspect(np.ptp(lim, axis=1)[[0, 2, 1]])
    ax.set_xlabel(f"x [{unidad}]")
    ax.set_ylabel(f"z (eje del tunel) [{unidad}]")
    ax.set_zlabel(f"y [{unidad}]")
    ax.view_init(*vista)
    ax.set_title(titulo)
    if apoyos is not None or presiones:
        ax.legend(loc="upper left", fontsize=8)
    return fig, ax
