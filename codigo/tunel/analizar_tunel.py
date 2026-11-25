import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "funciones"))

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.ticker import MaxNLocator
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

from etapa1_malla.elementos import caras_borde
from etapa1_malla.malla_tunel import PAREDES
from etapa4_resolver.tensiones import von_mises


# Analiza los resultados que guardo resolver_tunel.py (no resuelve nada: solo lee el .npz).
#   python3 analizar_tunel.py                  -> el .npz mas reciente de resultados/
#   python3 analizar_tunel.py tunel_H20.npz    -> ese archivo
# Imprime equilibrio, desplazamientos, von Mises y fuerzas en los pernos, y guarda 3 figuras en
# resultados/: <nombre>_3d.png, <nombre>_seccion.png y <nombre>_desarrollo.png

fy = 345e6                  # fluencia ASTM A572 Gr 50 [Pa]
radio_apoyo = 50e-3         # los maximos "lejos de los apoyos" excluyen esta distancia a pernos y extremos [m]
amplificacion = None        # de la deformada; None = automatica


# --- Leer ---------------------------------------------------------------------------------------
carpeta = Path(__file__).parent / "resultados"
if len(sys.argv) > 1:
    archivo = Path(sys.argv[1])
    archivo = archivo if archivo.exists() else carpeta / archivo
else:
    archivo = max(carpeta.glob("*.npz"), key=lambda f: f.stat().st_mtime)
r = np.load(archivo)
d = json.loads(str(r["datos"]))

xy, conectividad, U, F, R = r["xy"], r["conectividad"], r["U"], r["F"], r["R"]
sigma = r["sigma_nodos"]
pernos_xyz, pernos_sz, pernos_fuerza, pernos_normal = (r[k] for k in ("pernos_xyz", "pernos_sz",
                                                                       "pernos_fuerza", "pernos_normal"))
s_esq, z_uniones = r["s_esquinas"], r["z_uniones"]
a, b, t, L = d["a"], d["b"], d["t"], d["L"]
mm, MPa, kN = 1e3, 1e-6, 1e-3


# --- Geometria del contorno ----------------------------------------------------------------------
## Pared (0 derecha, 1 techo, 2 izquierda, 3 piso), s a lo largo del contorno, s local en la pared,
## distancia hacia afuera desde la cara interior (0 = cara interior, t = exterior) y normal exterior
def contorno(P):
    x, y = P[:, 0], P[:, 1]
    dx, dy = np.abs(x) - a/2, np.abs(y) - b/2
    lado_x = dx >= dy
    w = np.where(lado_x, np.where(x > 0, 0, 2), np.where(y > 0, 1, 3))
    s_loc = _s_local(w, x, y)
    n = np.zeros((len(P), 3))
    n[lado_x, 0] = np.sign(x[lado_x])
    n[~lado_x, 1] = np.sign(y[~lado_x])
    return w, s_esq[w] + s_loc, s_loc, np.maximum(dx, dy), n


def _s_local(w, x, y):
    xc, yc = np.clip(x, -a/2, a/2), np.clip(y, -b/2, b/2)
    return np.choose(w, [yc + b/2, a/2 - xc, b/2 - yc, xc + a/2])


## Nodos de la pared w incluyendo las dos esquinas (para los mapas del desarrollo)
def en_pared(w, P, tol=1e-9):
    x, y = P[:, 0], P[:, 1]
    dx, dy = np.abs(x) - a/2, np.abs(y) - b/2
    if w in (0, 2):
        return (dx >= dy - tol) & (np.sign(x) == (1 if w == 0 else -1))
    return (dy >= dx - tol) & (np.sign(y) == (1 if w == 1 else -1))


pared, s, s_loc, prof, normal = contorno(xy)
z = xy[:, 2]
interior = np.isclose(prof, 0, atol=1e-9)
exterior = np.isclose(prof, t, atol=1e-9)
un = -(U*normal).sum(axis=1)                        # desplazamiento normal, + hacia adentro del tunel
u_abs = np.linalg.norm(U, axis=1)
vm = von_mises(sigma)
vm_elem_max = von_mises(r["sigma_elem"]).max()

dist_perno = cKDTree(pernos_xyz).query(xy)[0]
lejos = dist_perno > radio_apoyo
if d["extremos_fijos"]:
    lejos &= (z > radio_apoyo) & (z < L - radio_apoyo)

traccion = (pernos_fuerza*pernos_normal).sum(axis=1)
pared_perno = contorno(pernos_xyz)[0]


def donde(n):
    cara = "interior" if prof[n] < t/2 else "exterior"
    return f"pared {PAREDES[pared[n]]}, s = {s_loc[n]*mm:.0f} mm, z = {z[n]*mm:.0f} mm, cara {cara}"


# --- Resumen -------------------------------------------------------------------------------------
print(f"{d['titulo']}  ({archivo.name}, resuelto {d['fecha']})")
planos = [p for p, v in (("x = 0", d.get("simetria")), ("z = L/2", d.get("simetria_z"))) if v]
if planos:
    print(f"  resuelto con simetria en {' y '.join(planos)}: {d['n_gdl_resuelto']} GDL "
          f"(1/{2**len(planos)} del tunel), reflejado al tunel completo")
print(f"  malla {d['tipo']}: {d['n_nodos']} nodos, {d['n_elem']} elementos, {d['n_gdl']} GDL, "
      f"{len(pernos_xyz)} pernos, extremos fijos: {d['extremos_fijos']}")
print(f"  t = {t*mm:.0f} mm, presion {d.get('carga', 'hidrostatica')}: {d['p_radier']*MPa:.4f} MPa (radier) a "
      f"{d['p_techo']*MPa:.4f} MPa (techo)")

print("\nEquilibrio [kN]               Fx         Fy         Fz")
print(f"  presion (suma F)    {F.sum(axis=0)[0]*kN:10.2f} {F.sum(axis=0)[1]*kN:10.2f} {F.sum(axis=0)[2]*kN:10.2f}")
print(f"  apoyos (suma R)     {R.sum(axis=0)[0]*kN:10.2f} {R.sum(axis=0)[1]*kN:10.2f} {R.sum(axis=0)[2]*kN:10.2f}")
print(f"  F + R               {(F + R).sum(axis=0)[0]*kN:10.2e} {(F + R).sum(axis=0)[1]*kN:10.2e} "
      f"{(F + R).sum(axis=0)[2]*kN:10.2e}   (tiene que ser ~0)")
# Empuje total hacia adentro = int p dA. Con signo (no |F|): en H20 las fuerzas consistentes de las
# esquinas de cada cara son negativas. En una pared la fuerza es solo normal, asi que sumar las dos
# componentes tambien cuenta bien los nodos de las esquinas del tunel (que reciben de dos paredes)
F_presion = -(F[:, 0]*np.sign(xy[:, 0]) + F[:, 1]*np.sign(xy[:, 1])).sum()
print(f"  la presion empuja {F_presion*kN:.0f} kN en total: pernos {traccion.sum()*kN:.0f} kN "
      f"({traccion.sum()/F_presion:.0%}), extremos {(F_presion - traccion.sum())*kN:.0f} kN")

print("\nDesplazamientos")
n = np.argmax(u_abs)
print(f"  |u| max = {u_abs[n]*mm:.4f} mm   ({donde(n)})")
n = np.argmax(np.where(interior, un, -np.inf))
print(f"  u normal max hacia adentro = {un[n]*mm:.4f} mm   ({donde(n)})")

print(f"\nTensiones de von Mises (nodales promediadas), fy = {fy*MPa:.0f} MPa")
n = np.argmax(vm)
print(f"  max global              {vm[n]*MPa:7.1f} MPa   ({donde(n)}, a {dist_perno[n]*mm:.0f} mm de un perno)")
print(f"  max en un elemento      {vm_elem_max*MPa:7.1f} MPa   (sin promediar)")
n_l = np.flatnonzero(lejos)[np.argmax(vm[lejos])]
vm_lejos = vm[n_l]
print(f"  max lejos de apoyos     {vm_lejos*MPa:7.1f} MPa   ({donde(n_l)})   FS = fy / sigma = {fy/vm_lejos:.2f}")
print(f"  (lejos = a mas de {radio_apoyo*mm:.0f} mm de pernos y extremos fijos. En un apoyo puntual la tension")
print("   es singular: el maximo global crece al refinar la malla y no es un valor de diseno)")
print("  max de von Mises segun la distancia excluida alrededor de los apoyos:")
for radio in (25e-3, 50e-3, 75e-3, 100e-3):
    sel = (dist_perno > radio) & ((z > radio) & (z < L - radio) if d["extremos_fijos"] else True)
    print(f"    > {radio*mm:3.0f} mm: {vm[sel].max()*MPa:6.1f} MPa")
print("  -> muy cerca del perno crece sin limite (apoyo de un nodo); desde unos 50 mm se estabiliza: es la")
print("     flexion de la plancha sobre los pernos (momento negativo). Para afinarlo usar d_perno > 0 (parche)")

print(f"\nPor pared                 u_n max   VM centro paño   VM int   VM ext   pernos  traccion max  prom")
print(f"                                    (en la u_n max)   (lejos de apoyos)")
for w in range(4):
    e = pared == w
    n_c = np.flatnonzero(e & interior)[np.argmax(un[e & interior])]
    print(f"  {PAREDES[w]:10s}            {un[n_c]*mm:7.4f} mm{vm[n_c]*MPa:11.1f} MPa"
          f"{vm[e & interior & lejos].max()*MPa:11.1f}{vm[e & exterior & lejos].max()*MPa:9.1f} MPa"
          f"{(pared_perno == w).sum():6d}{traccion[pared_perno == w].max()*kN:11.2f} kN"
          f"{traccion[pared_perno == w].mean()*kN:7.2f} kN")
k = np.argmax(traccion)
print(f"\nPernos: traccion max {traccion[k]*kN:.2f} kN (pared {PAREDES[pared_perno[k]]}, "
      f"s = {pernos_sz[k, 0]*mm:.0f} mm, z = {pernos_sz[k, 1]*mm:.0f} mm), min {traccion.min()*kN:.2f} kN, "
      f"promedio {traccion.mean()*kN:.2f} kN")
print(f"  referencia: p_radier x paso_s x paso_z = "
      f"{d['p_radier']*d['paso_s']*d['paso_z']*kN:.2f} kN (area tributaria con el paso maximo: cota superior)")


# --- Figuras ------------------------------------------------------------------------------------
escala = amplificacion or 0.025*max(a, b)/u_abs.max()
vm_max = vm_lejos                                   # escala de color: lo de mas (cerca de apoyos) satura
salida = lambda sufijo: carpeta / f"{archivo.stem}_{sufijo}.png"
G = lambda P: P[..., [0, 2, 1]]*mm                  # (x, y, z) modelo -> (x, z, y) grafico: y hacia arriba


# 1) 3D: deformada coloreada por |u| y por von Mises
caras = np.array([c[:4] for c in caras_borde(conectividad)])
def_xyz = xy + escala*U
fig = plt.figure(figsize=(15, 9), layout="constrained")
fig.suptitle(f"{d['titulo']}: deformada x{escala:.0f}  ({d['tipo']}, {d['n_gdl']} GDL)")
for j, (valor, norm, cmap, etiqueta) in enumerate([
        (u_abs*mm, colors.Normalize(0, u_abs.max()*mm), "viridis", "|u| [mm]"),
        (vm*MPa, colors.Normalize(0, vm_max*MPa), "YlOrRd", "von Mises [MPa] (satura cerca de apoyos)")]):
    ax = fig.add_subplot(1, 2, 1 + j, projection="3d")
    ax.add_collection3d(Poly3DCollection(G(def_xyz[caras]), facecolors=plt.get_cmap(cmap)(norm(valor[caras].mean(axis=1))),
                                         edgecolors="0.3", linewidths=0.05))
    lim = np.c_[G(def_xyz).min(axis=0), G(def_xyz).max(axis=0)]
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
    ax.set_box_aspect(np.ptp(lim, axis=1))
    ax.set_xlabel("x [mm]"); ax.set_ylabel("z (eje del tunel) [mm]"); ax.set_zlabel("y [mm]")
    ax.view_init(20, -50)
    fig.colorbar(cm.ScalarMappable(norm, cmap), ax=ax, shrink=0.5, pad=0.0, label=etiqueta)
    ax.set_title(["Desplazamiento", "Tension de von Mises (nodal)"][j])
fig.savefig(salida("3d"), dpi=150)


# 2) Seccion en el z de la flecha maxima: deformada y valores a lo largo del contorno
n_max = np.argmax(np.where(interior & lejos, un, -np.inf))
z_sec = z[n_max]
en_sec = np.isclose(z, z_sec)
fila = np.argmin(np.abs(np.unique(pernos_sz[:, 1]) - z_sec))
z_pernos = np.unique(pernos_sz[:, 1])[fila]
pernos_fila = np.isclose(pernos_sz[:, 1], z_pernos)

fig = plt.figure(figsize=(17, 9), layout="constrained")
fig.suptitle(f"Seccion z = {z_sec*mm:.0f} mm (donde la flecha es maxima)")
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.6])
ax = fig.add_subplot(gs[:, 0])
for cara, estilo in ((interior, "-"), (exterior, "--")):
    sel = np.flatnonzero(en_sec & cara)
    sel = np.r_[sel[np.argsort(s[sel])], sel[np.argsort(s[sel])][:1]]       # cerrado
    ax.plot(xy[sel, 0]*mm, xy[sel, 1]*mm, estilo, color="0.6", lw=1)
    ax.plot(def_xyz[sel, 0]*mm, def_xyz[sel, 1]*mm, estilo, color="#1a5fb4", lw=1.5)
ax.plot(pernos_xyz[pernos_fila, 0]*mm, pernos_xyz[pernos_fila, 1]*mm, "o", color="#c01c28", ms=4,
        label=f"pernos (fila z = {z_pernos*mm:.0f} mm)")
ax.plot([], [], color="0.6", label="sin deformar"); ax.plot([], [], color="#1a5fb4", label=f"deformada x{escala:.0f}")
ax.plot([], [], "k--", lw=1, label="cara exterior")
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="center", fontsize=9)
ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
ax.set_title("Contorno interior (-) y exterior (--)")

ax_u, ax_s = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
for ax in (ax_u, ax_s):
    for s_e in s_esq:
        ax.axvline(s_e*mm, color="k", lw=0.8)
    for sp in np.unique(pernos_sz[:, 0]):
        ax.axvline(sp*mm, color="#c01c28", lw=0.4, alpha=0.4)
    ax.set_xlim(0, s_esq[-1]*mm)
    ax.grid(alpha=0.3)
for w in range(4):
    ax_u.text((s_esq[w] + s_esq[w + 1])/2*mm, 1.02, PAREDES[w], transform=ax_u.get_xaxis_transform(), ha="center")
sel = np.flatnonzero(en_sec & interior)
sel = sel[np.argsort(s[sel])]
ax_u.plot(s[sel]*mm, un[sel]*mm, color="#1a5fb4")
ax_u.set_ylabel("u normal hacia adentro [mm]")
for cara, color, nombre in ((interior, "#e66100", "cara interior"), (exterior, "#5e3c99", "cara exterior")):
    sel = np.flatnonzero(en_sec & cara)
    sel = sel[np.argsort(s[sel])]
    ax_s.plot(s[sel]*mm, vm[sel]*MPa, color=color, label=nombre)
ax_s.set_ylabel("von Mises [MPa]")
ax_s.set_xlabel("s a lo largo del contorno [mm] (rojo: posiciones de los pernos en s)")
ax_s.legend()
fig.savefig(salida("seccion"), dpi=150)


# 3) Desarrollo (contorno desenrollado): mapas por cara y fuerza en los pernos
filas = [("u normal hacia adentro, cara interior [mm]", interior, un*mm, colors.Normalize(min(0, un[interior].min()*mm), un[interior].max()*mm), "viridis", "neither"),
         ("von Mises cara interior [MPa]", interior, vm*MPa, colors.Normalize(0, vm_max*MPa), "YlOrRd", "max"),
         ("von Mises cara exterior [MPa]", exterior, vm*MPa, colors.Normalize(0, vm_max*MPa), "YlOrRd", "max")]
fig, axs = plt.subplots(len(filas) + 1, 1, figsize=(18, 11), layout="constrained")
fig.suptitle(f"Desarrollo del blindaje (contorno desenrollado) - {d['tipo']}, von Mises saturado en "
             f"{vm_max*MPa:.0f} MPa (max lejos de apoyos)")
for ax, (titulo, cara, valor, norm, cmap, extender) in zip(axs, filas):
    niveles = np.linspace(norm.vmin, norm.vmax, 21)
    for w in range(4):
        sel = cara & en_pared(w, xy)
        sx = s_esq[w] + _s_local(np.full(sel.sum(), w), xy[sel, 0], xy[sel, 1])
        cs = ax.tricontourf(Triangulation(sx*mm, z[sel]*mm), valor[sel], levels=niveles, cmap=cmap,
                            norm=norm, extend=extender)
    fig.colorbar(cs, ax=ax, fraction=0.025, pad=0.01, ticks=MaxNLocator(6))
    ax.set_title(titulo, fontsize=10)
fs = axs[-1].scatter(pernos_sz[:, 0]*mm, pernos_sz[:, 1]*mm, c=traccion*kN, cmap="plasma", s=25, edgecolors="k",
                     linewidths=0.3, zorder=3)
fig.colorbar(fs, ax=axs[-1], fraction=0.025, pad=0.01)
axs[-1].set_title("Traccion en los pernos [kN]", fontsize=10)
for i, ax in enumerate(axs):
    for s_e in s_esq:
        ax.axvline(s_e*mm, color="k", lw=1)
    for zu in z_uniones:
        ax.axhline(zu*mm, color="0.3", lw=0.8, ls="--")
    if i < len(filas):
        ax.plot(pernos_sz[:, 0]*mm, pernos_sz[:, 1]*mm, ".", color="k", ms=2)
    ax.set_xlim(0, s_esq[-1]*mm); ax.set_ylim(0, L*mm)
    ax.set_aspect("equal")
    ax.set_ylabel("z [mm]")
    ax.set_xticks(np.r_[s_esq, (s_esq[:-1] + s_esq[1:])/2]*mm)
    ax.set_xticklabels([f"{v*mm:.0f}" for v in s_esq] + PAREDES, fontsize=8)
axs[-1].set_xlabel("s a lo largo del contorno interior [mm]  (lineas punteadas: uniones de planchas)")
fig.savefig(salida("desarrollo"), dpi=150)

print(f"\nFiguras: {', '.join(salida(k).name for k in ('3d', 'seccion', 'desarrollo'))} en {carpeta}")
plt.show()
