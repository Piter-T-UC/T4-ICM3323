import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "funciones"))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from unidades import ureg
from etapa1_malla.mallador import caja, nodos_en, apoyos_en, presion_en
from etapa1_malla.elementos import caras_borde
from casos_3d import resolver


# Viga cuadrada maciza en voladizo: misma viga resuelta de dos formas
#   1) solido 3D con hexaedros (cada nodo tiene ux, uy, uz)
#   2) elementos de viga Euler-Bernoulli (cada nodo tiene w y theta, funciones de Hermite)
# Empotrada en x = 0, carga repartida q (presion) sobre la cara superior z = a -> w = q a [N/m]
titulo = "Voladizo de seccion cuadrada maciza"

E = 200 * ureg.GPa
nu = 0.3
L = 1000 * ureg.mm              # largo
a = 100 * ureg.mm               # lado de la seccion (L/a = 10)
q = 2 * ureg.MPa                # presion sobre la cara superior

tipo = "H20"
n_x, n_s = 20, 2                # hexaedros en el largo y por lado de la seccion
n_viga = 4                      # elementos de viga


L_m, a_m = L.to(ureg.m).magnitude, a.to(ureg.m).magnitude
E_Pa = E.to(ureg.Pa).magnitude
w = -q.to(ureg.Pa).magnitude*a_m        # carga por unidad de largo [N/m], hacia -z
I = a_m**4/12
A = a_m**2


# --- 1) Solido 3D -----------------------------------------------------------------------------
x, y, z = np.linspace(0, L_m, n_x + 1), np.linspace(0, a_m, n_s + 1), np.linspace(0, a_m, n_s + 1)
xy_m, conectividad = caja(x, y, z, tipo)
xy = (xy_m*ureg.m).to(ureg.mm)
apoyos = np.array(apoyos_en(xy, lambda x, y, z: np.isclose(x, 0), 1, 1, 1), dtype=int)
presiones = presion_en(xy, conectividad, lambda x, y, z: np.isclose(z, a_m), q)
U, sigma, tiempo = resolver(xy, conectividad, E, nu, apoyos, presiones)

# Flecha del solido en cada seccion x: promedio de uz de todos los nodos de esa seccion
x_3d = np.unique(np.round(xy_m[:, 0], 12))
w_3d = np.array([U[nodos_en(xy, lambda x, y, z, xi=xi: np.isclose(x, xi)), 2].mean() for xi in x_3d])


# --- 2) Elementos de viga Euler-Bernoulli -----------------------------------------------------
## Viga de n_e elementos de Hermite, empotrada en x = 0, carga repartida w [N/m]
## Devuelve x de los nodos y d = [w0, th0, w1, th1, ...]
def viga_hermite(L, EI, w, n_e):
    l = L/n_e
    k = EI/l**3*np.array([[ 12,    6*l,  -12,    6*l],
                          [6*l, 4*l**2, -6*l, 2*l**2],
                          [-12,   -6*l,   12,   -6*l],
                          [6*l, 2*l**2, -6*l, 4*l**2]])
    f = w*l/2*np.array([1, l/6, 1, -l/6])           # cargas nodales consistentes: wl/2, wl^2/12
    n_gdl = 2*(n_e + 1)
    K, F = np.zeros((n_gdl, n_gdl)), np.zeros(n_gdl)
    for e in range(n_e):
        g = slice(2*e, 2*e + 4)
        K[g, g] += k
        F[g] += f
    libres = np.arange(2, n_gdl)                    # empotramiento: w0 = th0 = 0
    d = np.zeros(n_gdl)
    d[libres] = np.linalg.solve(K[np.ix_(libres, libres)], F[libres])
    return np.linspace(0, L, n_e + 1), d


## Flecha dentro de cada elemento con las funciones de Hermite (curva suave entre nodos)
def interpolar_hermite(x_n, d, m=30):
    xs, ws = [], []
    s = np.linspace(0, 1, m)
    for e in range(len(x_n) - 1):
        l = x_n[e + 1] - x_n[e]
        H = np.column_stack([1 - 3*s**2 + 2*s**3, l*(s - 2*s**2 + s**3), 3*s**2 - 2*s**3, l*(s**3 - s**2)])
        xs.append(x_n[e] + s*l)
        ws.append(H @ d[2*e:2*e + 4])
    return np.concatenate(xs), np.concatenate(ws)


x_viga, d_viga = viga_hermite(L_m, E_Pa*I, w, n_viga)
xs_viga, ws_viga = interpolar_hermite(x_viga, d_viga)

# Teoria: Euler-Bernoulli y Timoshenko (agrega la deformacion por corte, kappa = 5/6)
G = E_Pa/(2*(1 + nu))
w_eb = lambda x: w*x**2*(6*L_m**2 - 4*L_m*x + x**2)/(24*E_Pa*I)
w_timo = lambda x: w_eb(x) + w*(L_m*x - x**2/2)/(5/6*G*A)


# --- Resultados -------------------------------------------------------------------------------
print(f"{titulo}: L = {L:~}, seccion {a:~} x {a:~}, q = {q:~} (w = {-w/1e3:.1f} kN/m)")
print(f"  solido {tipo}: {n_x} x {n_s} x {n_s} elementos, {len(xy)} nodos, {3*len(xy)} GDL [{tiempo:.2f} s]")
print(f"  viga: {n_viga} elementos de Hermite, {2*(n_viga + 1)} GDL\n")
print("  flecha en la punta:")
print(f"    solido 3D           {-w_3d[-1]*1e3:.4f} mm")
print(f"    elementos de viga   {-d_viga[-2]*1e3:.4f} mm")
print(f"    Euler-Bernoulli     {-w_eb(L_m)*1e3:.4f} mm  (w L^4 / 8 E I)")
print(f"    Timoshenko          {-w_timo(L_m)*1e3:.4f} mm  (+ corte)")
print(f"\n  solido / viga = {w_3d[-1]/d_viga[-2]:.4f}")
print("  Los 4 elementos de viga dan exacto Euler-Bernoulli en los nodos (Hermite + cargas consistentes).")
print(f"  El solido tiene dos efectos que la viga no ve y que casi se cancelan:")
print(f"    + deformacion por corte (Timoshenko: {w_timo(L_m)/w_eb(L_m) - 1:+.1%})")
print("    - el empotramiento fija ux = uy = uz en toda la cara x = 0: impide la contraccion de")
print("      Poisson en la raiz y la rigidiza")


# --- Grafico ----------------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 6), layout="constrained")
fig.suptitle(titulo)
mm = 1e3
escala = 0.15*L_m/abs(w_3d).max()               # amplificacion para que se vea la deformada

# Izquierda: solido deformado (color = flecha) y eje de la viga deformado
ax = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)    # la linea roja va encima
caras = caras_borde(conectividad)
esquinas = np.array([cara[:4] for cara in caras])
def_m = xy_m + escala*U
ax.add_collection3d(Poly3DCollection(xy_m[esquinas]*mm, facecolors=(0, 0, 0, 0), edgecolors="0.75", linewidths=0.3))
color = plt.get_cmap("viridis_r")((-U[esquinas, 2].mean(axis=1))/abs(w_3d).max())
ax.add_collection3d(Poly3DCollection(def_m[esquinas]*mm, facecolors=color, edgecolors="0.3", linewidths=0.2))
ax.plot(xs_viga*mm, np.full_like(xs_viga, a_m/2)*mm, (a_m/2 + escala*ws_viga)*mm, color="#c01c28", lw=2.5,
        label="eje de la viga (elementos de viga)", zorder=10)
ax.set_yticks([0, a_m*mm])
ax.set_xlim(0, L_m*mm)
ax.set_ylim(0, a_m*mm)
z_min = min(0, def_m[:, 2].min())*mm
ax.set_zlim(z_min, a_m*mm)
ax.set_box_aspect((L_m*mm, a_m*mm, a_m*mm - z_min))
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_zlabel("z [mm]")
ax.view_init(18, -60)
ax.set_title(f"Solido 3D {tipo} deformado (x{escala:.0f}), gris = sin deformar")
ax.legend(loc="upper left", fontsize=8)

# Derecha: flecha a lo largo de la viga
ax = fig.add_subplot(1, 2, 2)
xt = np.linspace(0, L_m, 200)
ax.plot(xt*mm, -w_timo(xt)*mm, color="0.6", lw=1, ls="--", label="Timoshenko (teoria)")
ax.plot(xs_viga*mm, -ws_viga*mm, color="#c01c28", lw=2, label=f"Elementos de viga ({n_viga}, Hermite)")
ax.plot(x_viga*mm, -d_viga[0::2]*mm, "o", color="#c01c28", ms=7, mfc="white", mew=2)
ax.plot(x_3d*mm, -w_3d*mm, "s", color="#1a5fb4", ms=4, label=f"Solido 3D {tipo} (promedio de la seccion)")
ax.set_xlabel("x [mm]")
ax.set_ylabel("flecha -uz [mm]")
ax.set_title("Flecha a lo largo de la viga")
ax.grid(alpha=0.3)
ax.legend()

fig.savefig(Path(__file__).parent / "viga_cuadrada.png", dpi=150)
plt.show()
