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
MPa = ureg.MPa


titulo = "Blindaje de acero antes/despues de la compuerta"
# Datos del proyecto (Voith 1530-VOI-320-0-CC-0072, cap. 8 y planos 2MHS08-0100/0101)

# Material: ASTM A572 Gr 50
E = 205_940 * MPa                # = 2.1e6 kgf/cm2 del informe
nu = 0.3
rho = 7850 * ureg.kg/ureg.m**3

# Geometria del tunel (hueco interior) y de las planchas
a = 2300 * mm                   # ancho interior
b = 4600 * mm                   # alto interior
L = 1500 * mm                   # largo del tramo (aguas arriba y aguas abajo: 1500, entre compuertas: 1420)
t = 18 * mm                     # espesor de calculo: 20 mm - 2 mm de corrosion

# ---- Carga ----------------------------------------------------------------------------------
# Filtraciones por detras del blindaje con el tunel vacio: presion sobre la cara exterior, empuja
# hacia adentro del tunel.
#   "hidrostatica": p = gamma_w (nivel_agua - cota), crece hacia abajo. Con estos datos da
#                   0.3235 MPa en el radier y 0.2784 MPa en el techo (si el nivel queda bajo
#                   el techo, arriba del nivel p = 0)
#   "constante":    p = p_constante en todo el contorno (lo que hace el informe: el valor del radier)
# Para otra distribucion, editar la funcion p(x, y, z) mas abajo
carga = "hidrostatica"
nivel_agua = 111.0 * ureg.m             # cota del nivel de diseno (Q1000) [m s.n.m.]
cota_radier = 78.01 * ureg.m            # cota del piso interior del conducto [m s.n.m.]
gamma_w = 9.80665 * ureg.kN/ureg.m**3   # peso especifico del agua (rho g)
p_constante = 0.3235 * MPa              # solo para carga = "constante" (3.299 kgf/cm2 del informe)

# ---- Variables de diseno --------------------------------------------------------------------
ancho_plancha = 750 * mm        # ancho de cada plancha en z (la ultima se queda con lo que sobre)
borde = 125 * mm                # distancia de los pernos al borde de la plancha (informe: <= 125 mm)
paso_s = 500 * mm               # separacion maxima entre pernos en el contorno (informe: <= 300 mm)
paso_z = 500 * mm               # separacion maxima entre pernos a lo largo del tunel
# Estudio de espaciamiento (H27, hidrostatica, simetria). Admisibles del informe (sec. 4.1):
# plancha 2400 kgf/cm2 = 235 MPa; perno limitado por corte del hormigon T/(pi 10 15 cm2) <= 15 kgf/cm2
# -> T <= 69.3 kN (vastago 73.9 kN, soldadura 72.7 kN). En z, con planchas de 750 y borde 125, el
# espaciamiento solo puede ser 250 (paso_z < 500) o 500 mm
# a) malla gruesa h = 100, n_t = 1, perno de un nodo (las tensiones de la plancha NO convergen ahi)
#   paso   pernos   flecha    VM plancha       perno max
#   300     288     0.10 mm    28 MPa (12 %)   25.6 kN (37 %)
#   400     228     0.26 mm    43 MPa (18 %)   33.8 kN (49 %)
#   500     128     1.09 mm   145 MPa (61 %)   59.5 kN (86 %)
#   600     112     1.31 mm   175 MPa (74 %)   66.7 kN (96 %)
# b) h = 50, n_t = 1, d_perno = 20 mm (parche del perno M20). VM centro = en la flecha maxima;
#    VM > 75 mm = maximo a mas de 75 mm de pernos y extremos (pegado al parche es singular)
#   paso   flecha    VM centro   VM > 75 mm        perno max
#   300    0.10 mm    19 MPa      27 MPa (12 %)    24.9 kN (36 %)
#   500    1.05 mm    64 MPa     129 MPa (55 %)    59.3 kN (86 %)
#   La fuerza en los pernos ya converge (h = 100 y 50 dan lo mismo); la flecha todavia sube ~12 %
#   de h = 100 a h = 50
# ---------------------------------------------------------------------------------------------

# ---- Malla: ACA SE REFINA --------------------------------------------------------------------
# Las lineas de la malla pasan siempre por las esquinas, las uniones de planchas y los pernos, y cada
# tramo entre ellas se parte en partes iguales de largo <= h (por eso con h = 100 mm los elementos
# quedan de 62 a 98 mm: el paso de los pernos, 290 y 250 mm, se parte en 3 o 4)
#
#   h        tamano maximo de elemento en el contorno (s) y a lo largo del tunel (z). Es el principal:
#            achicarlo refina TODA la plancha
#   n_t      elementos en el espesor. Con H20 / H27 basta 1 para la flexion; 2 mejora poco y cuesta caro
#   tipo     "H8" | "H20" | "H27". H8 se traba en flexion con 1 elemento en el espesor (verificar_viga.py)
#   d_perno  refinamiento LOCAL en los pernos: > 0 agrega lineas a +-d/2 de cada perno y el perno fija un
#            parche de d x d (impide el giro, como la cabeza de 100 mm del informe). 0 = un nodo (rotula)
#
# Tamano del problema con esta geometria (GDL del tunel completo y de la mitad, con simetria = True).
# Medido con el solver directo: 55 mil GDL ~11 s, 111 mil GDL ~47 s y 3.4 GB de memoria; la memoria
# crece mas rapido que los GDL, asi que lo de mas de ~200 mil GDL no esta probado
#   tipo   h [mm]  n_t  d_perno    completo      mitad
#   H20    100     1     0          65 250      32 934
#   H27    100     1     0         110 700      55 719
#   H27     75     1     0         169 344      85 113
#   H27    100     2     0         184 500      92 865
#   H27    100     1    20         287 820     144 495
#   H27     50     1     0         336 960     169 065
#   H27     50     2     0         561 600     281 775
tipo = "H27"
h = 50 * mm
n_t = 2
d_perno = 0 * mm
# Pernos M20 soldados a la cara exterior: ux = uy = uz = 0 en el perno, giro libre (rotula), como en
# el informe. En un solido los nodos no tienen giros, asi que fijar UN nodo por perno ya deja girar.
# ---------------------------------------------------------------------------------------------

# Bordes z = 0 y z = L: el informe supone que donde una plancha toca a otra no hay desplazamiento
extremos_fijos = True

# Simetria en el plano x = 0: geometria, pernos y carga (que solo depende de y) son iguales a ambos
# lados, asi que se resuelve la mitad x >= 0 con ux = 0 en el plano (la mitad de los GDL) y
# resolver_tunel.py refleja los resultados: el .npz queda con el tunel completo.
# En y NO hay simetria (el radier tiene mas presion que el techo)
simetria = True


# --- Malla -----------------------------------------------------------------------------------
m = lambda v: v.to(ureg.m).magnitude
pernos = pernos_grilla(m(a), m(b), m(L), m(ancho_plancha), m(borde), m(paso_s), m(paso_z))
# Otra distribucion: cualquier arreglo (n, 2) de posiciones (s, z) en metros, por ejemplo
# pernos = np.array([(s, z) for ...])

xy_m, conectividad, info = malla_tunel(m(a), m(b), m(L), m(t), m(ancho_plancha), pernos,
                                       m(h), n_t, tipo, m(d_perno), mitad=simetria)
xy = (xy_m * ureg.m).to(mm)
validar_malla(xy, conectividad)

# Apoyos [nodo, rx, ry, rz]: nodos de cada perno con ux = uy = uz = 0, los extremos del tramo si
# extremos_fijos y, con simetria, ux = 0 en el plano x = 0 (un nodo puede juntar varias)
r = np.zeros((len(xy), 3), dtype=int)
r[[n for nodos in nodos_pernos(xy, info) for n in nodos]] = 1
if extremos_fijos:
    r[nodos_en(xy, lambda x, y, z: np.isclose(z, 0) or np.isclose(z, m(L)))] = 1
if simetria:
    r[nodos_en(xy, lambda x, y, z: np.isclose(x, 0)), 0] = 1
filas = np.flatnonzero(r.any(axis=1))
apoyos = np.column_stack([filas, r[filas]])

# Presion en un punto (x, y, z) [m] de la cara exterior, segun 'carga'. y se mide desde el centro
# del hueco, asi que la cota de un punto es cota_radier + (y + b/2)
def cota(y):
    return cota_radier + (y + m(b)/2)*ureg.m

def p(x, y, z):
    if carga == "hidrostatica":
        return max(gamma_w*(nivel_agua - cota(y)), 0*gamma_w*ureg.m).to(MPa)
    if carga == "constante":
        return p_constante.to(MPa)
    raise ValueError(f"carga '{carga}' no existe (opciones: 'hidrostatica', 'constante')")

presiones = presion_en(xy, conectividad, en_exterior(info), p)


# Al importar este archivo (resolver_tunel.py) solo se arma el modelo; el resumen y los graficos
# de la malla salen al correrlo directo
if __name__ == "__main__":
    # --- Resumen ---------------------------------------------------------------------------------
    n_planchas = len(uniones_planchas(m(L), m(ancho_plancha))) - 1
    volumen = ((a + 2*t)*(b + 2*t) - a*b)*L
    F = gen_f_presion(xy, presiones)
    print(f"{titulo}")
    p_piso, p_techo = p(0, -m(b)/2, 0), p(0, m(b)/2, 0)        # en la cara interior
    print(f"  tunel {a:~} x {b:~}, L = {L:~}, t = {t:~}")
    print(f"  presion {carga}: {p_piso:.4f~} en el radier, {p_techo:.4f~} en el techo")
    print(f"  {n_planchas} plancha(s) de {ancho_plancha:~} por pared (4 paredes), "
          f"{len(pernos)} pernos (borde {borde:~}, paso {paso_s:~} x {paso_z:~})")
    print(f"  peso del acero: {(rho*volumen).to(ureg.kg):.0f~}")
    print(f"  malla {tipo}{' (MITAD x >= 0, simetria)' if simetria else ''}: {len(xy)} nodos, "
          f"{len(conectividad)} elementos, {3*len(xy)} GDL, {len(apoyos)} nodos con apoyo, "
          f"{len(presiones)} caras con presion")
    # Revision: en la pared derecha p es lineal en y -> la fuerza total es la presion a media altura
    # (y = 0) por el area exterior (b + 2t) L
    F_teo = (p(0, 0, 0)*(b + 2*t)*L).to(ureg.kN).magnitude
    print(f"  revision: Fx en la pared derecha = {F[0::3][xy_m[:, 0] > 0].sum()/1e3:.1f} kN "
          f"(p(media altura) (b + 2t) L = {-F_teo:.1f} kN)")
    # Empuje neto hacia arriba: diferencia de presion entre las caras exteriores del piso y del techo
    dp = p(0, -m(b)/2 - m(t), 0) - p(0, m(b)/2 + m(t), 0)
    ancho = (a/2 + t) if simetria else (a + 2*t)
    print(f"            Fy neto (empuje hacia arriba) = {F[1::3].sum()/1e3:.1f} kN "
          f"(dp {'(a/2 + t)' if simetria else '(a + 2t)'} L = {(dp*ancho*L).to(ureg.kN).magnitude:.1f} kN)")


    # --- Graficos --------------------------------------------------------------------------------
    fig, axs = graficar_desarrollo(info, titulo=titulo)
    fig.savefig(Path(__file__).parent / "desarrollo_tunel.png", dpi=150)

    fig, ax = graficar_malla(xy, conectividad, apoyos, presiones, color_elem=info["plancha"],
                             titulo=f"{titulo} - malla {tipo} (colores: planchas)")
    fig.savefig(Path(__file__).parent / "malla_tunel.png", dpi=150)
    plt.show()
