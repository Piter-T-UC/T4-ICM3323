import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "funciones"))

import json
import time
from datetime import datetime

import numpy as np

import problema_tunel as P          # datos, malla, apoyos y presiones (sin graficar)
from unidades import ureg
from etapa1_malla.malla_tunel import nodos_pernos
from etapa3_matrices.matriz_D import matriz_D
from etapa3_matrices.gen_matriz_dispersa import gen_matriz_dispersa
from etapa3_matrices.gen_f_presion import gen_f_presion
from etapa4_resolver.restricciones import Restriciones
from etapa4_resolver.sistema_disperso import resolver_disperso
from etapa4_resolver.tensiones import calc_tensiones
from etapa4_resolver.simetria import (reflejar_malla, reflejar_campo, reflejar_fuerza, espejo_u,
                                      espejo_sigma)


# Resuelve el blindaje armado en problema_tunel.py y guarda todo en un .npz (floats SI: m, N, Pa)
# para analizarlo despues con analizar_tunel.py sin volver a resolver.
#   python3 resolver_tunel.py            -> resultados/tunel_<tipo>_<carga>.npz
#   python3 resolver_tunel.py otro       -> resultados/otro.npz
# Para cambiar el problema (carga, malla, pernos, espesor...) se editan los datos de problema_tunel.py.

nombre = sys.argv[1] if len(sys.argv) > 1 else f"tunel_{P.tipo}_{P.carga}_h{P.m(P.h)*1e3:.0f}_nt{P.n_t}"
archivo = Path(__file__).parent / "resultados" / f"{nombre}.npz"

m = P.m
info = P.info
reloj = {}

print(f"{P.titulo}: malla {P.tipo}, {len(P.xy)} nodos, {len(P.conectividad)} elementos, {3*len(P.xy)} GDL")
print(f"  presion {P.carga}: {P.p(0, -m(P.b)/2, 0):.3f~} en el radier, {P.p(0, m(P.b)/2, 0):.3f~} en el techo")

inicio = time.perf_counter()
D = matriz_D(P.E, P.nu, "3D")
K = gen_matriz_dispersa(P.xy, P.conectividad, D)
F = gen_f_presion(P.xy, P.presiones)
reloj["ensamble"] = time.perf_counter() - inicio
print(f"  K ensamblada: {K.nnz} no ceros [{reloj['ensamble']:.1f} s]")

inicio = time.perf_counter()
U = resolver_disperso(K, F, Restriciones(P.apoyos))
reloj["solucion"] = time.perf_counter() - inicio
print(f"  K U = F resuelto [{reloj['solucion']:.1f} s]")

inicio = time.perf_counter()
sigma_nodos, sigma_elem = calc_tensiones(P.xy, P.conectividad, U, D)
reloj["tensiones"] = time.perf_counter() - inicio
print(f"  tensiones [{reloj['tensiones']:.1f} s]")

# Reacciones R = K U - F: fuerza que los apoyos hacen sobre el acero (0 en los GDL libres)
R = K @ U - F

# Fuerza de cada perno = suma de las reacciones de sus nodos. Traccion = componente a lo largo de la
# normal exterior (el perno tira la plancha hacia el hormigon)
a, b, t = info["a"], info["b"], info["t"]
nodos_p = nodos_pernos(P.xy, info)
pxyz = info["pernos_xyz"]
normal = np.zeros_like(pxyz)
en_x = np.isclose(np.abs(pxyz[:, 0]), a/2 + t)
normal[en_x, 0] = np.sign(pxyz[en_x, 0])
normal[~en_x, 1] = np.sign(pxyz[~en_x, 1])
R3 = R.reshape(-1, 3)
fuerza_perno = np.array([R3[nodos].sum(axis=0) for nodos in nodos_p])

# Resultados a guardar (de la malla resuelta)
xy_g, conec_g, apoyos_g = P.xy_m, np.asarray(P.conectividad), P.apoyos
U_g, F_g, R_g, sn_g, se_g = U.reshape(-1, 3), F.reshape(-1, 3), R3, sigma_nodos, np.array(sigma_elem)
pared_g, plancha_g, sz_g = info["pared"], info["plancha"], info["pernos_sz"]

# Con simetria se resolvio solo un lado de cada plano (x >= 0 y/o z <= L/2): se refleja en cada plano
# para guardar el tunel completo. Se refleja primero en z = L/2 y despues en x = 0
perimetro = info["s_esquinas"][-1]
n_planchas = len(info["z_uniones"]) - 1
planos = ([(2, m(P.L)/2)] if P.simetria_z else []) + ([(0, 0.0)] if P.simetria else [])
for eje, valor in planos:
    eu, es = espejo_u(eje), espejo_sigma(eje)
    xy_g, conec_g, espejo, plano, otros = reflejar_malla(xy_g, conec_g, eje, valor)
    U_g = reflejar_campo(U_g, plano, otros, eu)
    sn_g = reflejar_campo(sn_g, plano, otros, es)
    se_g = np.concatenate([se_g, se_g*es])
    F_g, R_g = reflejar_fuerza(F_g, plano, otros, eu), reflejar_fuerza(R_g, plano, otros, eu)
    if eje == 0:
        pared_g = np.r_[pared_g, np.array([2, 1, 0, 3])[pared_g]]       # derecha <-> izquierda
        plancha_g = np.r_[plancha_g, plancha_g]
    else:
        pared_g = np.r_[pared_g, pared_g]
        plancha_g = np.r_[plancha_g, n_planchas - 1 - plancha_g]        # z -> L - z invierte las planchas
    lado = ~plano[apoyos_g[:, 0]]
    apoyos_g = np.vstack([apoyos_g, np.column_stack([espejo[apoyos_g[lado, 0]], apoyos_g[lado, 1:]])])
    # pernos: los del plano (si hay) suman las dos mitades; los demas se duplican reflejados.
    en_plano = np.isclose(pxyz[:, eje], valor, atol=1e-9)
    fuerza_perno[en_plano] += fuerza_perno[en_plano]*eu
    pxyz_e = pxyz[~en_plano].copy()
    pxyz_e[:, eje] = 2*valor - pxyz_e[:, eje]
    sz_e = sz_g[~en_plano].copy()
    if eje == 0:     # s del reflejado: s' = (2b + a - s) mod perimetro (derecha <-> izquierda, techo y piso al reves)
        sz_e[:, 0] = (2*b + a - sz_e[:, 0]) % perimetro
    else:            # el reflejado en z tiene el mismo s y z' = L - z
        sz_e[:, 1] = m(P.L) - sz_e[:, 1]
    pxyz = np.vstack([pxyz, pxyz_e])
    normal = np.vstack([normal, normal[~en_plano]*eu])
    fuerza_perno = np.vstack([fuerza_perno, fuerza_perno[~en_plano]*eu])
    sz_g = np.vstack([sz_g, sz_e])
    nodos_p = nodos_p + [list(espejo[nodos]) for nodos, e in zip(nodos_p, en_plano) if not e]
if planos:
    print(f"  simetria ({'x = 0' if P.simetria else ''}{' y ' if P.simetria and P.simetria_z else ''}"
          f"{'z = L/2' if P.simetria_z else ''}): resuelta 1/{2**len(planos)} ({3*len(P.xy)} GDL), "
          f"reflejada a {len(xy_g)} nodos")

datos = dict(
    titulo=P.titulo, fecha=datetime.now().isoformat(timespec="seconds"), tipo=P.tipo,
    E=P.E.to(ureg.Pa).magnitude, nu=P.nu, a=m(P.a), b=m(P.b), L=m(P.L), t=m(P.t),
    ancho_plancha=m(P.ancho_plancha), borde=m(P.borde), paso_s=m(P.paso_s), paso_z=m(P.paso_z),
    d_perno=m(P.d_perno), h=m(P.h), n_t=P.n_t, extremos_fijos=P.extremos_fijos, carga=P.carga,
    p_radier=P.p(0, -m(P.b)/2, 0).to(ureg.Pa).magnitude, p_techo=P.p(0, m(P.b)/2, 0).to(ureg.Pa).magnitude,
    simetria=P.simetria, simetria_z=P.simetria_z, perno_cara=P.perno_cara, n_gdl_resuelto=3*len(P.xy),          # GDL del sistema que se resolvio
    n_nodos=len(xy_g), n_elem=len(conec_g), n_gdl=3*len(xy_g), tiempos=reloj,   # del tunel completo
)

archivo.parent.mkdir(exist_ok=True)
np.savez_compressed(
    archivo,
    datos=json.dumps(datos),
    # malla (siempre el tunel completo, aunque se haya resuelto la mitad)
    xy=xy_g, conectividad=conec_g, pared=pared_g, plancha=plancha_g,
    s_esquinas=info["s_esquinas"], z_uniones=info["z_uniones"], apoyos=apoyos_g,
    # resultados por nodo (n, 3) / (n, 6) y por elemento (n_elem, n_nodos_e, 6) sin promediar
    U=U_g, F=F_g, R=R_g, sigma_nodos=sn_g, sigma_elem=se_g,
    # pernos: posicion, normal exterior, fuerza (3) y sus nodos (lista plana + inicio de cada perno)
    pernos_xyz=pxyz, pernos_sz=sz_g, pernos_normal=normal, pernos_fuerza=fuerza_perno,
    pernos_nodos=np.concatenate(nodos_p), pernos_inicio=np.cumsum([0] + [len(n) for n in nodos_p]),
)
print(f"\nGuardado: {archivo} ({archivo.stat().st_size/1e6:.1f} MB)")
print(f"Para analizarlo: python3 analizar_tunel.py {archivo.name}")
