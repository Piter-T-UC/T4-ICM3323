import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "funciones"))

import numpy as np

from unidades import ureg
# 1. malla
from etapa1_malla.elementos import ELEMENTOS, caras
# 2. interpolacion
from etapa2_interpolacion.funciones_forma import funciones_forma, NODOS_NATURALES
# 3. matrices
from etapa3_matrices.matriz_B import matriz_B
from etapa3_matrices.matriz_D import matriz_D
from etapa3_matrices.puntos_gauss import puntos_gauss
from etapa3_matrices.gen_matriz_dispersa import k_elemento
from etapa3_matrices.gen_f_presion import f_eq_cara

np.set_printoptions(precision=4, suppress=True, linewidth=160)


# Revisiones de los hexaedros H8, H20, H27 (lo mismo que verificar_elemento.py de T3, en 3D)
E = 200 * ureg.GPa
nu = 0.3
D = matriz_D(E, nu, "3D")

# Elemento fisico [m]: nodos naturales pasados por un mapeo afin + algo de distorsion
distorsion = lambda p: np.c_[2 + 1.2*p[:, 0] + 0.3*p[:, 1] + 0.10*p[:, 0]*p[:, 2],
                             1 + 0.2*p[:, 0] + 0.9*p[:, 1] - 0.05*p[:, 2]**2,
                             0.5 + 0.1*p[:, 1] + 0.4*p[:, 2]]

# Desplazamientos lineales u = G x: la deformacion exacta es constante (patch test)
G = np.array([[0.10, 0.02, -0.01],
              [0.03, -0.05, 0.04],
              [0.00, 0.01, 0.07]])
eps_exacta = np.array([G[0, 0], G[1, 1], G[2, 2], G[0, 1] + G[1, 0], G[1, 2] + G[2, 1], G[2, 0] + G[0, 2]])
W = np.array([[0, -0.1, 0.05], [0.1, 0, -0.2], [-0.05, 0.2, 0]])       # rotacion rigida (antisimetrica)

pt = np.array([0.3, -0.4, 0.2])
for tipo in ELEMENTOS:
    nat = NODOS_NATURALES[tipo]
    xyz = distorsion(nat)
    N_nodos = np.array([funciones_forma(tipo, *p)[0] for p in nat])
    N, dN = funciones_forma(tipo, *pt)

    # derivadas por diferencias finitas
    h = 1e-6
    dN_num = np.array([(funciones_forma(tipo, *(pt + h*e))[0] - funciones_forma(tipo, *(pt - h*e))[0])/(2*h)
                       for e in np.eye(3)])

    B, detJ, _ = matriz_B(tipo, xyz, *pt)
    u_lin = (xyz @ G.T).ravel()
    u_rig = (xyz @ W.T).ravel()

    # volumen = int det(J) dxi deta dzeta (H8 no puede seguir el termino zeta^2: da otro volumen)
    pts, pesos = puntos_gauss(tipo)
    vol = sum(w*matriz_B(tipo, xyz, *p)[1] for p, w in zip(pts, pesos))

    K = k_elemento(xyz, D)
    autovalores = np.linalg.eigvalsh(K)
    n_cero = int(np.sum(np.abs(autovalores) < 1e-8*autovalores.max()))

    # presion uniforme sobre la cara zeta = +1: la fuerza total debe ser -p * (vector area)
    cara = caras(tipo)[5]
    f = f_eq_cara(xyz[cara], 1.0)

    print(f"\n=== {tipo} ({len(nat)} nodos, {len(pesos)} puntos de Gauss) ===")
    print(f"  N_i(nodo_j) = identidad:           {np.allclose(N_nodos, np.eye(len(nat)))}")
    print(f"  suma N = 1, suma dN = 0:            {np.isclose(N.sum(), 1)}, {np.allclose(dN.sum(axis=1), 0)}")
    print(f"  dN = diferencias finitas:           {np.allclose(dN, dN_num, atol=1e-7)}")
    print(f"  patch test  B u (u = G x) = eps:    {np.allclose(B @ u_lin, eps_exacta)}")
    print(f"  rotacion rigida -> eps = 0:         {np.allclose(B @ u_rig, 0)}")
    print(f"  volumen int det J =                 {vol:.6f} m3")
    print(f"  K simetrica:                        {np.allclose(K, K.T, rtol=1e-10, atol=1e-6*np.abs(K).max())}")
    print(f"  modos de energia nula de K:         {n_cero}   (deben ser 6: 3 traslaciones + 3 rotaciones)")
    print(f"  presion p = 1 Pa en cara zeta+ : suma f = {f.sum(axis=0)}")
