import numpy as np

from etapa2_interpolacion.funciones_forma import funciones_forma


## Jacobiano de la transformacion natural -> fisica de un elemento isoparametrico (igual que en T3,
## sirve en 2D y 3D)
## dN: (dim, n) derivadas naturales (funciones_forma), xy_e: (n, dim) coordenadas de los nodos [m]
##   J[i, j] = dx_j/dxi_i  = dN @ xy_e
## Devuelve J, det(J) y dN_dx (dim, n): fila i = dN/dx_i  (dN_dx = J^-1 dN)
def jacobiano(dN, xy_e):
    J = dN @ xy_e
    detJ = np.linalg.det(J)
    if detJ <= 0:
        raise ValueError(f"det(J) = {detJ:.3g} <= 0: elemento invertido o muy distorsionado "
                         f"(revisar orden de los nodos y nodos medios)")
    return J, detJ, np.linalg.solve(J, dN)


## Matriz B en el punto natural (xi, eta[, zeta]): eps = B u_e
##   2D: (3, 2n), eps = [exx, eyy, gxy],                 u_e = [u0x, u0y, u1x, ...]
##   3D: (6, 3n), eps = [exx, eyy, ezz, gxy, gyz, gzx],  u_e = [u0x, u0y, u0z, u1x, ...]
## (g = deformacion angular ingenieril = 2 e). La dimension sale de xy_e (n, dim).
## Devuelve B, det(J) y N (para integrar con Gauss: dV = det(J) dxi deta dzeta)
def matriz_B(tipo, xy_e, *natural):
    xy_e = np.asarray(xy_e, dtype=float)
    N, dN = funciones_forma(tipo, *natural)
    _, detJ, d = jacobiano(dN, xy_e)            # d[0] = dN/dx, d[1] = dN/dy, d[2] = dN/dz
    n, dim = len(N), xy_e.shape[1]

    if dim == 2:
        B = np.zeros((3, 2*n))
        B[0, 0::2] = d[0]       # exx = du/dx
        B[1, 1::2] = d[1]       # eyy = dv/dy
        B[2, 0::2] = d[1]       # gxy = du/dy + dv/dx
        B[2, 1::2] = d[0]
        return B, detJ, N

    B = np.zeros((6, 3*n))
    B[0, 0::3] = d[0]           # exx = du/dx
    B[1, 1::3] = d[1]           # eyy = dv/dy
    B[2, 2::3] = d[2]           # ezz = dw/dz
    B[3, 0::3] = d[1]           # gxy = du/dy + dv/dx
    B[3, 1::3] = d[0]
    B[4, 1::3] = d[2]           # gyz = dv/dz + dw/dy
    B[4, 2::3] = d[1]
    B[5, 2::3] = d[0]           # gzx = dw/dx + du/dz
    B[5, 0::3] = d[2]
    return B, detJ, N
