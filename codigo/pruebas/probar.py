import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "funciones"))

import numpy as np

from unidades import ureg
from etapa1_malla.elementos import ELEMENTOS, caras, caras_borde
from etapa1_malla.mallador import caja, nodos_en, apoyos_en, presion_en
from etapa2_interpolacion.funciones_forma import funciones_forma, NODOS_NATURALES
from etapa3_matrices.matriz_D import matriz_D
from etapa3_matrices.puntos_gauss import gauss_1d, puntos_gauss
from etapa3_matrices.matriz_B import jacobiano, matriz_B
from etapa3_matrices.gen_matriz_dispersa import k_elemento, gen_matriz_dispersa
from etapa3_matrices.gen_f_presion import f_eq_cara, gen_f_presion
from etapa4_resolver.restricciones import Restriciones
from etapa4_resolver.sistema_disperso import resolver_disperso
from etapa4_resolver.tensiones import calc_tensiones, von_mises

np.set_printoptions(precision=4, suppress=True, linewidth=150)


# Probar las funciones de a una, viendo que entra y que sale.
#   python probar.py              -> menu
#   python probar.py matriz_D     -> solo esa prueba
#   python probar.py todas        -> todas seguidas
# Para cambiar los datos, editar la prueba (cada una es corta e independiente).

E = 200 * ureg.GPa
nu = 0.3
tipo = "H8"
lados = np.array([2.0, 1.0, 0.5])                           # caja de 2 x 1 x 0.5 m
xyz = (NODOS_NATURALES[tipo] + 1)/2 * lados                 # nodos del elemento [m]


def titulo(texto):
    print(f"\n{'='*90}\n  {texto}\n{'='*90}")


def p_matriz_D():
    titulo(f"matriz_D(E = {E:~}, nu = {nu}, estado)")
    for estado in ("3D", "tension_plana", "deformacion_plana"):
        D = matriz_D(E, nu, estado)
        print(f"\n{estado}: forma {D.shape}, en GPa")
        print(D/1e9)
    D = matriz_D(E, nu, "3D")
    print(f"\nD[3, 3] = {D[3, 3]/1e9:.4f} GPa = G = E/2(1+nu) = {E.to('GPa').m/(2*(1 + nu)):.4f} GPa")
    print(f"simetrica: {np.allclose(D, D.T)},  autovalores > 0: {np.all(np.linalg.eigvalsh(D) > 0)}")
    eps = np.array([1e-3, 0, 0, 0, 0, 0])
    print(f"traccion pura eps = {eps} -> sigma = D eps = {D @ eps/1e6} MPa")


def p_puntos_gauss():
    titulo("gauss_1d(n) y puntos_gauss(tipo)")
    for n in (1, 2, 3):
        s, w = gauss_1d(n)
        print(f"\nn = {n}: puntos {s}, pesos {w}")
        for grado in range(2*n + 1):
            exacto = (1 - (-1)**(grado + 1))/(grado + 1)    # int_-1^1 x^grado dx
            print(f"   int x^{grado} = {exacto:.4f}   Gauss = {np.sum(w*s**grado):.4f}"
                  f"   {'ok' if np.isclose(exacto, np.sum(w*s**grado)) else 'NO (grado > 2n-1)'}")
    for t in ELEMENTOS:
        pts, w = puntos_gauss(t)
        print(f"\n{t}: {len(w)} puntos, suma de pesos = {w.sum():.4f} (volumen del cubo [-1,1]^3 = 8)")
    pts, w = puntos_gauss("H8")
    print("\nH8 [xi, eta, zeta] y peso:")
    print(np.c_[pts, w])


def p_funciones_forma():
    titulo(f"funciones_forma('{tipo}', xi, eta, zeta)")
    print("nodos naturales:")
    print(NODOS_NATURALES[tipo])
    for punto in [(0, 0, 0), (1, 1, 1), (0.5, -0.2, 0.3)]:
        N, dN = funciones_forma(tipo, *punto)
        print(f"\nen {punto}:  N = {N}")
        print(f"   suma N = {N.sum():.6f}   dN (filas d/dxi, d/deta, d/dzeta):")
        print(dN)


def p_matriz_B():
    titulo(f"jacobiano y matriz_B en un {tipo} de {lados[0]} x {lados[1]} x {lados[2]} m")
    print("nodos del elemento [m]:")
    print(xyz)
    _, dN = funciones_forma(tipo, 0, 0, 0)
    J, detJ, dN_dx = jacobiano(dN, xyz)
    print(f"\nJ en el centro (debe ser diag(lados/2) = {lados/2}):")
    print(J)
    print(f"det J = {detJ:.4f}   (volumen fisico / volumen natural = {lados.prod():.2f}/8 = {lados.prod()/8:.4f})")
    B, detJ, N = matriz_B(tipo, xyz, 0, 0, 0)
    print(f"\nB en el centro: forma {B.shape}  (6 deformaciones x 3 GDL por nodo)")
    print(B)
    u = (xyz*[1e-3, 0, 0]).ravel()                         # ux = 0.001 x -> exx = 0.001
    print(f"\nprueba: ux = 0.001 x  ->  eps = B u = {B @ u}")


def p_k_elemento():
    titulo(f"k_elemento: rigidez de un {tipo} de {lados[0]} x {lados[1]} x {lados[2]} m")
    K = k_elemento(xyz, matriz_D(E, nu, "3D"))
    print(f"forma {K.shape}, en GN/m. Primeras 6 filas/columnas (nodos 0 y 1):")
    print(K[:6, :6]/1e9)
    autoval = np.linalg.eigvalsh(K)
    print(f"\nsimetrica: {np.allclose(K, K.T, atol=1e-6*abs(K).max())}")
    print(f"autovalores ~ 0: {np.sum(abs(autoval) < 1e-8*autoval.max())} (6 = movimientos de cuerpo rigido)")
    print(f"suma de cada fila de ux (traslacion rigida en x no genera fuerza): {K[:, 0::3].sum(axis=1).max():.2e}")


def p_f_eq_cara():
    titulo("f_eq_cara: fuerzas nodales de una presion sobre una cara")
    cara = caras(tipo)[5]
    print(f"cara zeta+ del {tipo}, nodos locales {cara}:")
    print(xyz[cara])
    f = f_eq_cara(xyz[cara], 1e3)
    area = lados[0]*lados[1]
    print(f"\np = 1 kPa -> f por nodo [N] (fx, fy, fz):")
    print(f)
    print(f"suma = {f.sum(axis=0)}   (p A = {1e3*area:.0f} N hacia -z)")
    p_var = np.array([0, 2e3, 2e3, 0])                       # lineal en x: 0 a 2 kPa
    print(f"\npresion variable por nodo {p_var} Pa -> f:")
    print(f_eq_cara(xyz[cara], p_var))


def p_malla():
    titulo("caja: malla de 2 x 1 x 1 elementos H8")
    xy_m, conectividad = caja([0, 1, 2], [0, 1], [0, 0.5], "H8")
    print("nodos [m]:")
    for i, p in enumerate(xy_m):
        print(f"  {i:2d}: {p}")
    print("\nconectividad:")
    print(np.array(conectividad))
    borde = caras_borde(conectividad)
    print(f"\ncaras de borde: {len(borde)} (2 elementos x 6 caras - 2 compartidas = 10)")
    xy = xy_m*ureg.m
    print(f"nodos en x = 0: {nodos_en(xy, lambda x, y, z: np.isclose(x, 0))}")


def p_resolver():
    titulo("gen_matriz_dispersa + Restriciones + resolver_disperso + calc_tensiones")
    print("barra de 2 x 1 x 0.5 m (2 H8), x = 0 fija solo en ux (+ lo minimo para no tener cuerpo rigido),")
    print("presion -1 MPa (traccion) en x = 2  ->  sxx = 1 MPa y ux(2) = sigma L / E")
    xy_m, conectividad = caja([0, 1, 2], [0, 1], [0, 0.5], "H8")
    xy = xy_m*ureg.m
    D = matriz_D(E, nu, "3D")
    r = np.zeros((len(xy_m), 3), dtype=int)
    r[nodos_en(xy, lambda x, y, z: np.isclose(x, 0)), 0] = 1
    r[nodos_en(xy, lambda x, y, z: x == 0 and y == 0 and z == 0), :] = 1
    r[nodos_en(xy, lambda x, y, z: x == 0 and y == 1 and z == 0), 2] = 1
    r[nodos_en(xy, lambda x, y, z: x == 0 and y == 0 and z == 0.5), 1] = 1
    fijos = np.flatnonzero(r.any(axis=1))
    apoyos = np.column_stack([fijos, r[fijos]])
    print(f"\napoyos [nodo, rx, ry, rz]:\n{apoyos}")
    print(f"GDL fijos: {Restriciones(apoyos)}")

    K = gen_matriz_dispersa(xy, conectividad, D)
    presiones = presion_en(xy, conectividad, lambda x, y, z: np.isclose(x, 2), -1*ureg.MPa)
    F = gen_f_presion(xy, presiones)
    print(f"\nK: {K.shape}, {K.nnz} no ceros.  F (fx de cada nodo) [kN]: {F[0::3]/1e3}")
    U = resolver_disperso(K, F, Restriciones(apoyos))
    print(f"\nU [mm] (ux, uy, uz por nodo):\n{U.reshape(-1, 3)*1e3}")
    print(f"ux en x = 2: {U[0::3].max()*1e3:.6f} mm   (sigma L / E = {1e6*2/E.to('Pa').m*1e3:.6f} mm)")
    sigma, _ = calc_tensiones(xy, conectividad, U, D)
    print(f"\nsigma nodo 0 [MPa] (sxx, syy, szz, txy, tyz, tzx): {sigma[0]/1e6}")
    print(f"von Mises nodo 0: {von_mises(sigma[0])/1e6:.4f} MPa")


PRUEBAS = {
    "matriz_D": p_matriz_D,
    "puntos_gauss": p_puntos_gauss,
    "funciones_forma": p_funciones_forma,
    "matriz_B": p_matriz_B,
    "k_elemento": p_k_elemento,
    "f_eq_cara": p_f_eq_cara,
    "malla": p_malla,
    "resolver": p_resolver,
}


if __name__ == "__main__":
    nombre = sys.argv[1] if len(sys.argv) > 1 else None
    if nombre is None:
        print("Pruebas disponibles:")
        for i, n in enumerate(PRUEBAS, 1):
            print(f"  {i}. {n}")
        nombre = input("numero o nombre (o 'todas'): ").strip()
        if nombre.isdigit():
            nombre = list(PRUEBAS)[int(nombre) - 1]
    for n in (PRUEBAS if nombre == "todas" else [nombre]):
        PRUEBAS[n]()
