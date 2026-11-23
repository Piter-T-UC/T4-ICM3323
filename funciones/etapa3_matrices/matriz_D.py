import numpy as np

from unidades import ureg


## Matriz constitutiva D en Pa: sigma = D eps (elastico lineal isotropo)
##   "tension_plana", "deformacion_plana": (3x3), sigma = [sxx, syy, txy], eps = [exx, eyy, gxy]
##   "3D":                                 (6x6), sigma = [sxx, syy, szz, txy, tyz, tzx]
##                                                 eps   = [exx, eyy, ezz, gxy, gyz, gzx]
## (g = 2 e, deformacion angular ingenieril)
def matriz_D(E, nu, estado="3D"):
    E = E.to(ureg.Pa).magnitude
    if estado == "3D":
        c = (1 - 2*nu)/2                            # corte: c E/((1+nu)(1-2nu)) = E/(2(1+nu)) = G
        return E/((1 + nu)*(1 - 2*nu)) * np.array([[1 - nu, nu,     nu,     0, 0, 0],
                                                   [nu,     1 - nu, nu,     0, 0, 0],
                                                   [nu,     nu,     1 - nu, 0, 0, 0],
                                                   [0,      0,      0,      c, 0, 0],
                                                   [0,      0,      0,      0, c, 0],
                                                   [0,      0,      0,      0, 0, c]])
    if estado == "tension_plana":
        return E/(1 - nu**2) * np.array([[1,  nu, 0],
                                         [nu, 1,  0],
                                         [0,  0,  (1 - nu)/2]])
    if estado == "deformacion_plana":
        return E/((1 + nu)*(1 - 2*nu)) * np.array([[1 - nu, nu,     0],
                                                   [nu,     1 - nu, 0],
                                                   [0,      0,      (1 - 2*nu)/2]])
    raise ValueError(f"estado '{estado}' no existe (opciones: '3D', 'tension_plana', 'deformacion_plana')")
