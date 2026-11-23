import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "funciones"))

from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from unidades import ureg
from etapa1_malla.elementos import caras_borde, TIPO_CARA
from etapa1_malla.mallador import caja, nodos_en, presion_en
from etapa2_interpolacion.funciones_forma import funciones_forma
from casos_3d import resolver, _apoyos


# Comparacion 3D "como se deberia ver" vs "como se ve": dos problemas con solucion exacta de
# elasticidad en TODO el solido (no solo en un punto), resueltos con H8, H20 y H27.
#   1) Cubo a traccion uniaxial: u es lineal -> todos los elementos tienen que dar exacto (patch test)
#   2) Barra en flexion pura:    u es cuadratico (uz ~ x^2) -> H8 (trilineal) no lo puede seguir y se
#                                traba; H20 y H27 tienen los cuadraticos completos y dan exacto
# El desplazamiento MEF se dibuja interpolado con las funciones de forma (no solo en los nodos), asi
# que tambien se ve lo que pasa dentro de cada elemento.

E = 200 * ureg.GPa
nu = 0.3
s = 100 * ureg.MPa                  # traccion aplicada / sigma_xx en la fibra superior en flexion
tipos = ["H8", "H20", "H27"]

# Caso 1: cubo de lado a con n elementos por lado
a = 100 * ureg.mm
n_cubo = 2

# Caso 2: barra de largo L y seccion h x h. n_s tiene que ser par (hace falta un nodo en el
# centro de la seccion para el apoyo en z)
L = 400 * ureg.mm
h = 100 * ureg.mm
n_x, n_s = 4, 2

sub = 4                             # subdivisiones por cara para dibujar (interpolacion dentro del elemento)


E_Pa, s_Pa = E.to(ureg.Pa).magnitude, s.to(ureg.Pa).magnitude
a_m, L_m, h_m = (v.to(ureg.m).magnitude for v in (a, L, h))
assert n_s % 2 == 0, "n_s tiene que ser par"


# --- Casos -------------------------------------------------------------------------------------
## Cada caso: lineas de malla, apoyos [(en, rx, ry, rz)], cara cargada y presion p (p > 0 empuja hacia
## adentro, asi que traccion = p negativa), solucion exacta u(P) y sigma_xx(P) con P (n, 3) en m,
## y una medida resumen (componente promedio en la cara cargada)

# Traccion: sigma_xx = s, el resto 0.  x = 0, y = 0, z = 0 son planos de simetria (rodillos)
#   ux = s x / E,  uy = -nu s y / E,  uz = -nu s z / E
traccion = SimpleNamespace(
    nombre="Cubo a traccion uniaxial", archivo="cubo_traccion.png",
    lineas=[np.linspace(0, a_m, n_cubo + 1)]*3,
    apoyos=[(lambda x, y, z: np.isclose(x, 0), 1, 0, 0),
            (lambda x, y, z: np.isclose(y, 0), 0, 1, 0),
            (lambda x, y, z: np.isclose(z, 0), 0, 0, 1)],
    cargada=lambda x, y, z: np.isclose(x, a_m),
    p=-s,
    exacta=lambda P: np.c_[s_Pa/E_Pa*P[:, 0], -nu*s_Pa/E_Pa*P[:, 1], -nu*s_Pa/E_Pa*P[:, 2]],
    sxx=lambda P: np.full(len(P), s_Pa),
    medida=("alargamiento ux en x = a", 0),
    texto_apoyos="x=0: ux=0,  y=0: uy=0,  z=0: uz=0 (planos de simetria)",
)

# Flexion pura: sigma_xx = s z'/c lineal en la altura (traccion arriba), con c = h/2 e
# y' = y - c, z' = z - c medidos desde el eje. Con kappa = s / (E c) (Saint-Venant):
#   ux = kappa x z',  uy = -nu kappa y' z',  uz = -kappa/2 (x^2 + nu (z'^2 - y'^2))
# Apoyos (todos en x = 0, compatibles con la solucion exacta): ux = 0 en la cara, uy = 0 en la
# linea y' = 0, uz = 0 en el centro de la seccion
c = h_m/2
kappa = s_Pa/(E_Pa*c)
def _flexion(P):
    x, yp, zp = P[:, 0], P[:, 1] - c, P[:, 2] - c
    return np.c_[kappa*x*zp, -nu*kappa*yp*zp, -kappa/2*(x**2 + nu*(zp**2 - yp**2))]

flexion = SimpleNamespace(
    nombre="Barra en flexion pura", archivo="barra_flexion.png",
    lineas=[np.linspace(0, L_m, n_x + 1), np.linspace(0, h_m, n_s + 1), np.linspace(0, h_m, n_s + 1)],
    apoyos=[(lambda x, y, z: np.isclose(x, 0), 1, 0, 0),
            (lambda x, y, z: np.isclose(x, 0) and np.isclose(y, c), 0, 1, 0),
            (lambda x, y, z: np.isclose(x, 0) and np.isclose(y, c) and np.isclose(z, c), 0, 0, 1)],
    cargada=lambda x, y, z: np.isclose(x, L_m),
    p=lambda x, y, z: -s*(z - c)/c,
    exacta=_flexion,
    sxx=lambda P: s_Pa*(P[:, 2] - c)/c,
    medida=("flecha uz en x = L", 2),
    texto_apoyos="x=0: ux=0;  y=h/2: uy=0;  centro: uz=0",
)


# --- Resolver ----------------------------------------------------------------------------------
def resolver_caso(caso, tipo):
    xy_m, conectividad = caja(*caso.lineas, tipo)
    xy = (xy_m*ureg.m).to(ureg.mm)
    apoyos = _apoyos(xy, caso.apoyos)
    presiones = presion_en(xy, conectividad, caso.cargada, caso.p)
    U, sigma, tiempo = resolver(xy, conectividad, E, nu, apoyos, presiones)
    Ue = caso.exacta(xy_m)
    cargados = nodos_en(xy, caso.cargada)
    k = caso.medida[1]
    return SimpleNamespace(
        tipo=tipo, xy_m=xy_m, conectividad=conectividad, apoyos=apoyos, U=U, Ue=Ue, tiempo=tiempo,
        err_u=np.abs(U - Ue).max()/np.abs(Ue).max(),
        err_s=np.abs(sigma[:, 0] - caso.sxx(xy_m)).max()/s_Pa,
        err_otras=np.abs(sigma[:, 1:]).max()/s_Pa,
        medida=U[cargados, k].mean(), medida_e=Ue[cargados, k].mean())


# --- Graficos ----------------------------------------------------------------------------------
## Caras de borde subdivididas sub x sub con las funciones de forma de la cara (Q4 / Q8 / Q9):
## X (n_caras, sub+1, sub+1, 3) posicion sin deformar y Uh (igual) desplazamiento MEF interpolado
def superficie(xy_m, conectividad, U, sub):
    caras = caras_borde(conectividad)
    t = np.linspace(-1, 1, sub + 1)
    N = np.array([[funciones_forma(TIPO_CARA[len(caras[0])], xi, eta)[0] for xi in t] for eta in t])
    return np.array([N @ xy_m[cara] for cara in caras]), np.array([N @ U[cara] for cara in caras])


def cuadritos(X):            # (n_caras, m+1, m+1, 3) -> (n_caras m m, 4, 3) poligonos
    return np.stack([X[:, :-1, :-1], X[:, :-1, 1:], X[:, 1:, 1:], X[:, 1:, :-1]], axis=-2).reshape(-1, 4, 3)


def promedio(V):             # (n_caras, m+1, m+1) -> valor de cada cuadrito
    return ((V[:, :-1, :-1] + V[:, :-1, 1:] + V[:, 1:, 1:] + V[:, 1:, :-1])/4).ravel()


def bordes(X):               # contorno de cada cara (las aristas de los elementos, curvas si se deforman)
    return [linea for Xc in X for linea in (Xc[0], Xc[-1], Xc[:, 0], Xc[:, -1])]


def superficie_color(ax, X, valores, norm, cmap):
    ax.add_collection3d(Poly3DCollection(cuadritos(X)*1e3, facecolors=cmap(norm(promedio(valores))),
                                         edgecolors="none", zorder=1))
    ax.add_collection3d(Line3DCollection([b*1e3 for b in bordes(X)], colors="0.25", linewidths=0.4, zorder=2))


def lineas(ax, X, **kw):
    ax.add_collection3d(Line3DCollection([b*1e3 for b in bordes(X)], zorder=5, **kw))


def graficar_caso(caso, resultados):
    r0 = resultados[0]
    X0, _ = superficie(r0.xy_m, r0.conectividad, r0.U.reshape(-1, 3), sub)
    Ue0 = caso.exacta(X0.reshape(-1, 3)).reshape(X0.shape)
    tam = np.ptp(r0.xy_m, axis=0).max()
    escala = 0.15*tam/np.abs(Ue0).max()

    todo = np.concatenate([X0.reshape(-1, 3), (X0 + escala*Ue0).reshape(-1, 3)])*1e3
    lim = np.c_[todo.min(axis=0), todo.max(axis=0)]
    lim += np.c_[-1, 1]*0.04*np.ptp(lim, axis=1).max()

    norm_u = colors.Normalize(0, np.linalg.norm(Ue0, axis=-1).max()*1e3)
    cmap_u = plt.get_cmap("viridis")
    errores = []
    for r in resultados:
        X, Uh = superficie(r.xy_m, r.conectividad, r.U.reshape(-1, 3), sub)
        Ue = caso.exacta(X.reshape(-1, 3)).reshape(X.shape)
        errores.append((X, Uh, Ue, np.linalg.norm(Uh - Ue, axis=-1)/np.abs(Ue).max()*100))
    norm_e = colors.Normalize(0, max(1.0, max(e[3].max() for e in errores)))   # en %, al menos 0-1 %
    cmap_e = plt.get_cmap("Reds")

    fig = plt.figure(figsize=(18, 9), layout="constrained")
    fig.suptitle(f"{caso.nombre}: solucion exacta vs MEF   (E = {E:~}, nu = {nu}, s = {s:~}, "
                 f"deformada x{escala:.0f})")
    filas = [[fig.add_subplot(2, 1 + len(resultados), 1 + j + f*(1 + len(resultados)), projection="3d",
                              computed_zorder=False) for j in range(1 + len(resultados))] for f in range(2)]

    def ejes(ax, titulo):
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
        ax.set_box_aspect(np.ptp(lim, axis=1))
        ax.set_xlabel("x [mm]", fontsize=8); ax.set_ylabel("y [mm]", fontsize=8); ax.set_zlabel("z [mm]", fontsize=8)
        ax.tick_params(labelsize=7)
        for eje, d in zip((ax.xaxis, ax.yaxis, ax.zaxis), np.ptp(lim, axis=1)):
            eje.set_major_locator(MaxNLocator(max(2, int(6*d/np.ptp(lim, axis=1).max()))))
        ax.view_init(20, -60)
        ax.set_title(titulo, fontsize=10)

    # Arriba a la izquierda: el problema (malla, apoyos y cargas)
    ax = filas[0][0]
    ax.add_collection3d(Poly3DCollection(cuadritos(X0)*1e3, facecolors="#cfd8e3", edgecolors="none", alpha=0.6, zorder=1))
    lineas(ax, X0, colors="0.4", linewidths=0.5)
    P = r0.xy_m[r0.apoyos[:, 0]]*1e3
    for k, (marca, color, nombre) in enumerate([("o", "#c01c28", "ux = 0"), ("s", "#26a269", "uy = 0"),
                                                ("^", "#1a5fb4", "uz = 0")]):
        sel = r0.apoyos[:, 1 + k] == 1
        ax.scatter(*P[sel].T, marker=marca, s=[40, 22, 12][k], facecolors="none", edgecolors=color,
                   depthshade=False, label=nombre, zorder=6)
    # flechas de la traccion sobre la cara cargada (x = constante, normal +x): t = -p n
    x_c = r0.xy_m[nodos_en(r0.xy_m, caso.cargada), 0][0]
    yy, zz = np.meshgrid(np.linspace(*lim[1]/1e3, 7)[1:-1], np.linspace(*lim[2]/1e3, 7)[1:-1])
    yy, zz = yy.ravel(), zz.ravel()
    dentro = (yy > r0.xy_m[:, 1].min()) & (yy < r0.xy_m[:, 1].max()) & (zz > r0.xy_m[:, 2].min()) & (zz < r0.xy_m[:, 2].max())
    yy, zz = yy[dentro], zz[dentro]
    p = np.array([(caso.p(x_c, y, z) if callable(caso.p) else caso.p).to(ureg.Pa).magnitude for y, z in zip(yy, zz)])
    largo = -p/s_Pa*0.25*tam*1e3                                     # traccion en x, positiva hacia afuera
    base = np.where(largo >= 0, x_c*1e3, x_c*1e3 - largo)            # compresion: la punta llega a la cara
    ax.quiver(base, yy*1e3, zz*1e3, largo, 0*largo, 0*largo, color="#e66100", lw=1, arrow_length_ratio=0.3,
              label="traccion aplicada", zorder=7)
    ax.legend(loc="upper left", fontsize=7)
    ejes(ax, f"Problema\n{caso.texto_apoyos}")

    # Abajo a la izquierda: como se deberia ver (solucion exacta)
    ax = filas[1][0]
    lineas(ax, X0, colors="0.7", linewidths=0.5)
    superficie_color(ax, X0 + escala*Ue0, np.linalg.norm(Ue0, axis=-1)*1e3, norm_u, cmap_u)
    ejes(ax, "Como se deberia ver: solucion exacta\n(color = |u|)")

    # Columnas: cada elemento. Arriba MEF (color |u|) con la exacta en negro punteado; abajo el error
    for j, (r, (X, Uh, Ue, err)) in enumerate(zip(resultados, errores), start=1):
        ax = filas[0][j]
        lineas(ax, X, colors="0.7", linewidths=0.5)
        superficie_color(ax, X + escala*Uh, np.linalg.norm(Uh, axis=-1)*1e3, norm_u, cmap_u)
        lineas(ax, X + escala*Ue, colors="k", linewidths=0.9, linestyles="--")
        ejes(ax, f"Como se ve: MEF {r.tipo} ({len(r.xy_m)} nodos)\n"
                 f"{caso.medida[0]}: {r.medida/r.medida_e:.4f} x exacta")
        ax = filas[1][j]
        superficie_color(ax, X + escala*Uh, err, norm_e, cmap_e)
        ejes(ax, f"Error {r.tipo}: |u_MEF - u_exacta| / max|u_exacta|\nmaximo = {err.max():.2g} %")

    filas[0][1].legend(handles=[Line2D([], [], color="0.7", lw=1, label="sin deformar"),
                                Line2D([], [], color="k", lw=1, ls="--", label="exacta deformada")],
                       loc="upper left", fontsize=7)
    fig.colorbar(cm.ScalarMappable(norm_u, cmap_u), ax=filas[0], shrink=0.7, label="|u| [mm] (tambien la exacta)")
    fig.colorbar(cm.ScalarMappable(norm_e, cmap_e), ax=filas[1], shrink=0.7, label="error [% de max|u_exacta|]")
    fig.savefig(Path(__file__).parent / caso.archivo, dpi=150)
    return fig


# --- Correr ------------------------------------------------------------------------------------
for caso in (traccion, flexion):
    print(f"\n{caso.nombre}   ({caso.texto_apoyos})")
    print(f"  {'tipo':5s}{'nodos':>7s}{'err u':>11s}{'err sxx':>11s}{'otras sig':>11s}"
          f"   {caso.medida[0]} (MEF / exacta)")
    resultados = [resolver_caso(caso, tipo) for tipo in tipos]
    for r in resultados:
        print(f"  {r.tipo:5s}{len(r.xy_m):7d}{r.err_u:11.2e}{r.err_s:11.2e}{r.err_otras:11.2e}"
              f"   {r.medida*1e3:.6f} / {r.medida_e*1e3:.6f} mm = {r.medida/r.medida_e:.4f}")
    graficar_caso(caso, resultados)

print("\nerr u = max|u_MEF - u_exacta| / max|u_exacta| (en los nodos), err sxx y otras sig = error / s")
print("Traccion: u lineal -> los tres elementos dan exacto (patch test), error ~ 1e-13.")
print("Flexion: uz ~ x^2 -> H8 (trilineal) no lo puede representar y se traba (shear locking): queda mas")
print("rigido. H20 y H27 contienen los cuadraticos completos -> exacto.")
plt.show()
