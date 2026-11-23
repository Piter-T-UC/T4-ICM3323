import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "funciones"))

import numpy as np
import matplotlib.pyplot as plt

from unidades import ureg
from soluciones_placa import placa_pernos, voladizo, rigidez_placa
from casos_3d import resolver_pano, resolver_voladizo
from etapa5_graficar.graficar_comparacion import graficar_curvas

mm = ureg.mm
MPa = ureg.MPa


# Comparacion de los hexaedros H8 / H20 / H27 en problemas con solucion conocida, pensados para el
# blindaje: plancha delgada en flexion apoyada en pernos. Cambiar los datos de aca y volver a correr.

# Material y plancha (los del blindaje)
E = 205_940 * MPa                # = 2.1e6 kgf/cm2
nu = 0.3
t = 18 * mm
q = 0.3235 * MPa                 # = 3.299 kgf/cm2

# ---- Prueba 1: pano de plancha entre 4 pernos (grilla cuadrada de paso a) ----------------------
# Kirchhoff exacto (serie de Fourier) vs MEF en 1/4 de pano, refinando la malla
paso = 300 * mm
tipos = ["H8", "H20", "H27"]
capas = [1, 2]                  # elementos en el espesor
divisiones = [2, 4, 8, 12]      # elementos por lado del cuarto de pano (150 mm)
d_apoyo = 20 * mm               # parche fijo en el perno (en el pano interior el giro es 0 por simetria)

# ---- Prueba 2: perno como rotula (1 nodo) vs parche, refinando -------------------------------
tipo_perno = "H20"
capas_perno = 2
apoyos_perno = [0 * mm, 10 * mm, 20 * mm]
divisiones_perno = [2, 4, 8, 12, 16]

# ---- Prueba 3: voladizo con elementos distorsionados (trapecios) -------------------------------
L_v = 300 * mm
b_v = 50 * mm
nx_v, ny_v, nt_v = 6, 1, 1
distorsiones = [0.0, 0.1, 0.2, 0.3, 0.4]


m = lambda v: v.to(ureg.m).magnitude
E_Pa, q_Pa, t_m = E.to(ureg.Pa).magnitude, q.to(ureg.Pa).magnitude, m(t)


# --- Prueba 1 --------------------------------------------------------------------------------
w_teo, M_teo, s_teo = placa_pernos(q_Pa, m(paso), t_m, E_Pa, nu)
print(f"Prueba 1: pano entre pernos, paso {paso:~}, t = {t:~} (a/t = {m(paso)/t_m:.1f}), "
      f"q = {q.to(ureg.MPa):.4f~}, parche {d_apoyo:~}")
print(f"  Kirchhoff: w centro = {w_teo*1e3:.4f} mm, sigma centro = {s_teo/1e6:.2f} MPa "
      f"(w = {w_teo*rigidez_placa(E_Pa, nu, t_m)/(q_Pa*m(paso)**4):.5f} q a^4/D)")
print(f"  {'tipo':5s}{'capas':>6s}{'n':>4s}{'elem':>6s}{'GDL':>8s}{'w/w_teo':>10s}{'s/s_teo':>10s}{'t [s]':>8s}")
series = []
for tipo in tipos:
    for e, n_t in enumerate(capas):
        r = [resolver_pano(m(paso), t_m, q_Pa, E, nu, n, n_t, tipo, m(d_apoyo)) for n in divisiones]
        for n, ri in zip(divisiones, r):
            print(f"  {tipo:5s}{n_t:6d}{n:4d}{ri.n_elem:6d}{ri.n_gdl:8d}{ri.w/w_teo:10.3f}{ri.sxx/s_teo:10.3f}"
                  f"{ri.tiempo:8.2f}")
        series.append(dict(label=f"{tipo}, {n_t} capa(s)", tipo=tipo, estilo=e, x=[ri.n_gdl for ri in r],
                           ys=[[ri.w/w_teo for ri in r], [ri.sxx/s_teo for ri in r]]))
fig, _ = graficar_curvas(series, "GDL", ["w centro / Kirchhoff", "sigma_xx centro (abajo) / Kirchhoff"],
                         f"Pano entre pernos: paso {paso:~}, t = {t:~}, parche {d_apoyo:~}", logx=True)
fig.savefig(Path(__file__).parent / "comparacion_pano.png", dpi=150)


# --- Prueba 2 --------------------------------------------------------------------------------
print(f"\nPrueba 2: apoyo del perno, {tipo_perno} con {capas_perno} capas (w incluye lo que se hunde el apoyo)")
print(f"  {'apoyo':>8s}{'n':>4s}{'GDL':>8s}{'w/w_teo':>10s}{'s/s_teo':>10s}")
series = []
for e, d in enumerate(apoyos_perno):
    r = [resolver_pano(m(paso), t_m, q_Pa, E, nu, n, capas_perno, tipo_perno, m(d)) for n in divisiones_perno]
    for n, ri in zip(divisiones_perno, r):
        print(f"  {d.magnitude:6.0f}mm{n:4d}{ri.n_gdl:8d}{ri.w/w_teo:10.3f}{ri.sxx/s_teo:10.3f}")
    nombre = "rotula (1 nodo)" if d.magnitude == 0 else f"parche {d:~}"
    series.append(dict(label=nombre, tipo=tipo_perno, estilo=e, x=[ri.n_gdl for ri in r],
                       ys=[[ri.w/w_teo for ri in r], [ri.sxx/s_teo for ri in r]]))
fig, _ = graficar_curvas(series, "GDL", ["w centro / Kirchhoff", "sigma_xx centro (abajo) / Kirchhoff"],
                         f"Apoyo del perno: rotula vs parche ({tipo_perno}, {capas_perno} capas)", logx=True)
fig.savefig(Path(__file__).parent / "comparacion_perno.png", dpi=150)


# --- Prueba 3 --------------------------------------------------------------------------------
w_v, s_v = voladizo(q_Pa, m(L_v), m(b_v), t_m, E_Pa)
print(f"\nPrueba 3: voladizo L = {L_v:~}, b = {b_v:~}, t = {t:~}, {nx_v}x{ny_v}x{nt_v} elementos, distorsionado")
print(f"  Euler-Bernoulli: flecha = {w_v*1e3:.4f} mm, sigma raiz = {s_v/1e6:.2f} MPa")
print(f"  {'tipo':5s}{'distorsion':>11s}{'w/w_teo':>10s}{'s/s_teo':>10s}")
series = []
for tipo in tipos:
    r = [resolver_voladizo(m(L_v), m(b_v), t_m, q_Pa, E, nu, nx_v, ny_v, nt_v, tipo, d) for d in distorsiones]
    for d, ri in zip(distorsiones, r):
        print(f"  {tipo:5s}{d:11.2f}{ri.w/w_v:10.3f}{ri.sxx/s_v:10.3f}")
    series.append(dict(label=tipo, tipo=tipo, x=distorsiones,
                       ys=[[ri.w/w_v for ri in r], [ri.sxx/s_v for ri in r]]))
fig, _ = graficar_curvas(series, "distorsion (corrimiento / largo del elemento)",
                         ["flecha punta / Euler-Bernoulli", "sigma_xx raiz / Euler-Bernoulli"],
                         f"Voladizo {nx_v}x{ny_v}x{nt_v} con elementos trapezoidales")
fig.savefig(Path(__file__).parent / "comparacion_distorsion.png", dpi=150)
plt.show()
