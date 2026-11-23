import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "funciones"))

import time
import numpy as np

from unidades import ureg
# 1. malla
from etapa1_malla.mallador import hexas_estructurados, nodos_en, apoyos_en, presion_en
from etapa1_malla.elementos import validar_malla
# 3. matrices
from etapa3_matrices.matriz_D import matriz_D
from etapa3_matrices.gen_matriz_dispersa import gen_matriz_dispersa
from etapa3_matrices.gen_f_presion import gen_f_presion
# 4. resolver
from etapa4_resolver.restricciones import Restriciones
from etapa4_resolver.sistema_disperso import resolver_disperso
from etapa4_resolver.tensiones import calc_tensiones


# Viga en voladizo hecha con una plancha delgada, para ver si los hexaedros aguantan la flexion de
# placa con UN solo elemento en el espesor (lo que pasa en el blindaje del tunel).
# Empotrada en x = 0, presion q sobre la cara superior. Euler-Bernoulli:
#   flecha en la punta  = w L^4 / (8 E I),  w = q b         (b = ancho)
#   sigma_xx max (raiz) = M c / I,          M = w L^2 / 2
titulo = "Voladizo de plancha delgada"

E = 200 * ureg.GPa
nu = 0.3
L = 600 * ureg.mm
b = 100 * ureg.mm
t = 18 * ureg.mm
q = 0.3235 * ureg.MPa          # 3.299 kgf/cm2 (presion del informe de Voith)

n_x, n_y, n_t = 12, 2, 1        # elementos en largo, ancho y espesor (tamano 50 mm ~ el del tunel)

L_m, b_m, t_m = (v.to(ureg.m).magnitude for v in (L, b, t))
I = b_m*t_m**3/12
w = q.to(ureg.Pa).magnitude*b_m
flecha_teo = w*L_m**4/(8*E.to(ureg.Pa).magnitude*I)
sigma_teo = w*L_m**2/2*(t_m/2)/I
print(f"{titulo}: L = {L}, b = {b}, t = {t}, q = {q}  ({n_x} x {n_y} x {n_t} elementos)")
print(f"Euler-Bernoulli: flecha = {flecha_teo*1e3:.4f} mm, sigma_xx raiz = {sigma_teo/1e6:.2f} MPa\n")

D = matriz_D(E, nu, "3D")
x, y, z = (np.linspace(0, v, n + 1) for v, n in ((L_m, n_x), (b_m, n_y), (t_m, n_t)))
X = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)

for tipo in ("H8", "H20", "H27"):
    inicio = time.perf_counter()
    xy_m, conectividad, _ = hexas_estructurados(X, tipo)
    xy = (xy_m*ureg.m).to(ureg.mm)
    validar_malla(xy, conectividad)

    apoyos = np.array(apoyos_en(xy, lambda x, y, z: np.isclose(x, 0), 1, 1, 1), dtype=int)
    presiones = presion_en(xy, conectividad, lambda x, y, z: np.isclose(z, t_m), q)

    K = gen_matriz_dispersa(xy, conectividad, D)
    F = gen_f_presion(xy, presiones)
    U = resolver_disperso(K, F, Restriciones(apoyos))

    punta = nodos_en(xy, lambda x, y, z: np.isclose(x, L_m))
    flecha = -U[3*np.array(punta) + 2].mean()
    sigma, _ = calc_tensiones(xy, conectividad, U, D)
    raiz_arriba = nodos_en(xy, lambda x, y, z: np.isclose(x, 0) and np.isclose(z, t_m))
    sxx = sigma[raiz_arriba, 0].mean()
    print(f"  {tipo:4s}: {len(xy):5d} nodos  flecha = {flecha*1e3:.4f} mm ({flecha/flecha_teo:6.1%})"
          f"   sigma_xx raiz = {sxx/1e6:7.2f} MPa ({sxx/sigma_teo:6.1%})"
          f"   suma Fz = {F[2::3].sum():.1f} N (q b L = {w*L_m:.1f})   [{time.perf_counter() - inicio:.2f} s]")

print("\nH8 con un elemento en el espesor se traba en flexion (shear locking): da una flecha muy chica.")
print("H20 / H27 con un elemento en el espesor sirven para las planchas del blindaje.")
