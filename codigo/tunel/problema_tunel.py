import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "funciones"))

import numpy as np
import matplotlib.pyplot as plt

from unidades import ureg
# 1. malla
from etapa1_malla.malla_tunel import (malla_tunel, pernos_grilla, nodos_pernos, en_exterior,
                                      uniones_planchas)
from etapa1_malla.mallador import nodos_en, presion_en
from etapa1_malla.elementos import validar_malla
# 3. matrices
from etapa3_matrices.gen_f_presion import gen_f_presion
# 5. graficar
from etapa5_graficar.graficar_malla import graficar_malla
from etapa5_graficar.graficar_tunel import graficar_desarrollo

mm = ureg.mm
kgf_cm2 = ureg.kgf/ureg.cm**2


titulo = "Blindaje de acero antes/despues de la compuerta"
# Datos del proyecto (Voith 1530-VOI-320-0-CC-0072, cap. 8 y planos 2MHS08-0100/0101)

# Material: ASTM A572 Gr 50
E = 2.1e6 * kgf_cm2
nu = 0.3
rho = 7850 * ureg.kg/ureg.m**3

# Geometria del tunel (hueco interior) y de las planchas
a = 2300 * mm                   # ancho interior
b = 4600 * mm                   # alto interior
L = 1500 * mm                   # largo del tramo (aguas arriba y aguas abajo: 1500, entre compuertas: 1420)
t = 18 * mm                     # espesor de calculo: 20 mm - 2 mm de corrosion

# Carga: filtraciones por detras del blindaje con el tunel vacio, empuja hacia adentro.
# Presion hidrostatica con el nivel de diseno: p = gamma_w (nivel - cota). En el radier da
# 32.99 m.c.a. = 3.299 kgf/cm2 (el valor constante que usa el informe) y en el techo 2.839 kgf/cm2
gamma_w = 1000 * ureg.kgf/ureg.m**3     # peso especifico del agua
nivel_agua = 111.0 * ureg.m             # cota del nivel de diseno (Q1000) [m s.n.m.]
cota_radier = 78.01 * ureg.m            # cota del piso interior del conducto [m s.n.m.]

# ---- Variables de diseno --------------------------------------------------------------------
ancho_plancha = 750 * mm        # ancho de cada plancha en z (la ultima se queda con lo que sobre)
borde = 125 * mm                # distancia de los pernos al borde de la plancha (informe: <= 125 mm)
paso_s = 300 * mm               # separacion maxima entre pernos en el contorno (informe: <= 300 mm)
paso_z = 300 * mm               # separacion maxima entre pernos a lo largo del tunel
# ---------------------------------------------------------------------------------------------

# Pernos M20 soldados a la cara exterior: ux = uy = uz = 0 en el perno, giro libre (rotula), como en
# el informe. En un solido los nodos no tienen giros, asi que fijar UN nodo por perno ya deja girar.
# d_perno > 0 fija un parche de d x d nodos, que ademas impide el giro (empotramiento local)
d_perno = 0 * mm

# Malla
tipo = "H20"                    # H8 se traba en flexion con un elemento en el espesor (ver verificar_viga.py)
h = 100 * mm                    # tamano maximo de elemento en el contorno y en z
n_t = 1                         # elementos en el espesor

# Bordes z = 0 y z = L: el informe supone que donde una plancha toca a otra no hay desplazamiento
extremos_fijos = True


# --- Malla -----------------------------------------------------------------------------------
m = lambda v: v.to(ureg.m).magnitude
pernos = pernos_grilla(m(a), m(b), m(L), m(ancho_plancha), m(borde), m(paso_s), m(paso_z))
# Otra distribucion: cualquier arreglo (n, 2) de posiciones (s, z) en metros, por ejemplo
# pernos = np.array([(s, z) for ...])

xy_m, conectividad, info = malla_tunel(m(a), m(b), m(L), m(t), m(ancho_plancha), pernos,
                                       m(h), n_t, tipo, m(d_perno))
xy = (xy_m * ureg.m).to(mm)
validar_malla(xy, conectividad)

# Apoyos: nodos de cada perno con ux = uy = uz = 0 (+ los extremos del tramo si extremos_fijos)
fijos = {n for nodos in nodos_pernos(xy, info) for n in nodos}
if extremos_fijos:
    fijos |= set(nodos_en(xy, lambda x, y, z: np.isclose(z, 0) or np.isclose(z, m(L))))
apoyos = np.array([[n, 1, 1, 1] for n in sorted(fijos)], dtype=int)

# Presion hidrostatica en la cara exterior, hacia adentro del tunel. y [m] se mide desde el centro
# del hueco, asi que la cota de un punto es cota_radier + (y + b/2)
cota = lambda y: cota_radier + (y + m(b)/2)*ureg.m
p = lambda x, y, z: (gamma_w*(nivel_agua - cota(y))).to(kgf_cm2)
presiones = presion_en(xy, conectividad, en_exterior(info), p)


# --- Resumen ---------------------------------------------------------------------------------
n_planchas = len(uniones_planchas(m(L), m(ancho_plancha))) - 1
volumen = ((a + 2*t)*(b + 2*t) - a*b)*L
F = gen_f_presion(xy, presiones)
print(f"{titulo}")
p_piso, p_techo = p(0, -m(b)/2, 0), p(0, m(b)/2, 0)        # en la cara interior
print(f"  tunel {a:~} x {b:~}, L = {L:~}, t = {t:~}")
print(f"  presion hidrostatica: {p_piso:.3f~} en el radier, {p_techo:.3f~} en el techo "
      f"({p_piso.to(ureg.MPa):.4f~} a {p_techo.to(ureg.MPa):.4f~})")
print(f"  {n_planchas} plancha(s) de {ancho_plancha:~} por pared (4 paredes), "
      f"{len(pernos)} pernos (borde {borde:~}, paso {paso_s:~} x {paso_z:~})")
print(f"  peso del acero: {(rho*volumen).to(ureg.kg):.0f~}")
print(f"  malla {tipo}: {len(xy)} nodos, {len(conectividad)} elementos, {3*len(xy)} GDL, "
      f"{len(apoyos)} nodos fijos, {len(presiones)} caras con presion")
# Revision: en la pared derecha p es lineal en y -> la fuerza total es la presion a media altura
# (y = 0) por el area exterior (b + 2t) L
F_teo = (p(0, 0, 0)*(b + 2*t)*L).to(ureg.kN).magnitude
print(f"  revision: Fx en la pared derecha = {F[0::3][xy_m[:, 0] > 0].sum()/1e3:.1f} kN "
      f"(p(media altura) (b + 2t) L = {-F_teo:.1f} kN)")
# Empuje neto hacia arriba: diferencia de presion entre las caras exteriores del piso y del techo
dp = p(0, -m(b)/2 - m(t), 0) - p(0, m(b)/2 + m(t), 0)
print(f"            Fy neto (empuje hacia arriba) = {F[1::3].sum()/1e3:.1f} kN "
      f"(dp (a + 2t) L = {(dp*(a + 2*t)*L).to(ureg.kN).magnitude:.1f} kN)")


# --- Graficos --------------------------------------------------------------------------------
fig, axs = graficar_desarrollo(info, titulo=titulo)
fig.savefig(Path(__file__).parent / "desarrollo_tunel.png", dpi=150)

fig, ax = graficar_malla(xy, conectividad, apoyos, presiones, color_elem=info["plancha"],
                         titulo=f"{titulo} - malla {tipo} (colores: planchas)")
fig.savefig(Path(__file__).parent / "malla_tunel.png", dpi=150)
plt.show()
