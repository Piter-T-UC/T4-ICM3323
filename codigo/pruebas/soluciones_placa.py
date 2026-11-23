import numpy as np


## Soluciones analiticas para comparar los elementos 3D. Todo en floats SI (Pa, m).


## Rigidez a flexion de placa D = E t^3 / (12 (1 - nu^2)) [N m]
def rigidez_placa(E, nu, t):
    return E*t**3/(12*(1 - nu**2))


## Placa infinita (Kirchhoff) apoyada en una grilla cuadrada de pernos de paso a, con presion q
## uniforme. Es el pano de plancha entre 4 pernos del blindaje, lejos de los bordes.
## Serie de Fourier: la reaccion de los pernos es un peine de cargas puntuales -q a^2, entonces
##   w(x, y) = q a^4 / (16 pi^4 D) * sum_{(j,k) != 0} (1 - cos(2 pi (j x + k y)/a)) / (j^2 + k^2)^2
## con el perno en (0, 0). En el centro del pano (a/2, a/2) el coseno vale (-1)^(j + k).
## Devuelve w_centro [m], M_centro [N m/m] (Mx = My por simetria) y sigma_centro = 6 M / t^2 [Pa]
## (Timoshenko, Theory of plates and shells, flat slabs: w = 0.00581 q a^4 / D)
def placa_pernos(q, a, t, E, nu, N=400):
    j, k = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1), indexing="ij")
    r4 = (j**2 + k**2).astype(float)**2
    r4[N, N] = np.inf                                   # el termino (0, 0) no va
    s = (-1.0)**(j + k)
    D = rigidez_placa(E, nu, t)
    w = q*a**4/(16*np.pi**4*D)*np.sum((1 - s)/r4)
    kappa = -q*a**2/(4*np.pi**2*D)*np.sum(j**2*s/r4)    # curvatura -w_xx en el centro (= -w_yy)
    M = D*(1 + nu)*kappa                                # Mx = -D (w_xx + nu w_yy), traccion abajo
    return w, M, 6*M/t**2


## Voladizo de largo L, ancho b y espesor t con presion q en la cara superior (Euler-Bernoulli)
## Devuelve flecha en la punta w q b L^4 / (8 E I) [m] y sigma_xx en la raiz M c / I [Pa]
def voladizo(q, L, b, t, E):
    I = b*t**3/12
    w = q*b
    return w*L**4/(8*E*I), w*L**2/2*(t/2)/I
