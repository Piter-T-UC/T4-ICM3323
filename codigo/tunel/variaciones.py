import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "funciones"))

import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

from etapa4_resolver.tensiones import von_mises


# Barrido de variaciones del blindaje: espesor de plancha, espaciamiento de pernos y carga.
#   python3 variaciones.py            -> resuelve lo que falte y arma resultados/variaciones.csv
#   python3 variaciones.py --forzar   -> vuelve a resolver todo
# Cada caso corre resolver_tunel.py en un proceso aparte, con T4_OVERRIDE pisando los datos de
# problema_tunel.py (no se edita nada). Las metricas salen del .npz de cada caso.

carpeta = Path(__file__).parent / "resultados"
# Modelo de barrido: H27, h = 50 mm, 1 elemento en el espesor, CUARTO de tunel (simetria en x = 0 y
# z = L/2) y cada perno fija la cara completa de un hexaedro de 50 mm (9 nodos) en vez de un nodo
base = dict(h=50, n_t=1, d_perno=50, perno_cara=True, simetria_z=True, tipo="H27")

espesores = [12, 15, 18, 20, 25, 30]                    # mm (18 = el de calculo del informe: 20 - 2 de corrosion)
pasos = [300, 400, 500, 600]                            # mm, paso_s = paso_z
casos = {f"cq_t{t}_p{p}": dict(base, t=t, paso_s=p, paso_z=p) for t in espesores for p in pasos}
# misma geometria con presion constante (la del radier en todo el contorno, como hace el informe)
casos.update({f"cq_t{t}_p500_const": dict(base, t=t, paso_s=500, paso_z=500, carga="constante")
              for t in (12, 18, 25)})

admisible_plancha = 235e6       # [Pa] informe sec. 4.1
admisible_perno = 69.3e3        # [N]  corte del hormigon
fy = 345e6                      # [Pa] A572 Gr 50
# Radio de exclusion alrededor de cada perno para el "von Mises lejos de apoyos" [m]. Con el perno como
# cara de 50 mm el borde empotrado del parche es singular y a 50 mm del centro (25 mm del borde) todavia
# se nota: el maximo baja mucho hasta ~75 mm y desde 100 mm ya no cambia (con un nodo, 50 mm bastaba)
radio_apoyo = 100e-3
workers = 5


def resolver(nombre, datos):
    if (carpeta / f"{nombre}.npz").exists() and "--forzar" not in sys.argv:
        return nombre, "ya estaba"
    entorno = dict(os.environ, T4_OVERRIDE=json.dumps(datos))
    p = subprocess.run([sys.executable, str(Path(__file__).parent / "resolver_tunel.py"), nombre],
                       env=entorno, capture_output=True, text=True)
    (carpeta / f"{nombre}.log").write_text(p.stdout + p.stderr, encoding="utf-8")
    return nombre, "ok" if p.returncode == 0 else f"ERROR (ver {nombre}.log)"


def metricas(nombre):
    r = np.load(carpeta / f"{nombre}.npz")
    d = json.loads(str(r["datos"]))
    xy, U, sigma = r["xy"], r["U"], r["sigma_nodos"]
    a, b, t, L = d["a"], d["b"], d["t"], d["L"]
    dx, dy = np.abs(xy[:, 0]) - a/2, np.abs(xy[:, 1]) - b/2
    lado_x = dx >= dy
    normal = np.zeros((len(xy), 3))
    normal[lado_x, 0] = np.sign(xy[lado_x, 0])
    normal[~lado_x, 1] = np.sign(xy[~lado_x, 1])
    un = -(U*normal).sum(axis=1)                        # + hacia adentro del tunel
    interior = np.isclose(np.maximum(dx, dy), 0, atol=1e-9)
    vm = von_mises(sigma)
    lejos = cKDTree(r["pernos_xyz"]).query(xy)[0] > radio_apoyo
    lejos &= (xy[:, 2] > radio_apoyo) & (xy[:, 2] < L - radio_apoyo)
    traccion = (r["pernos_fuerza"]*r["pernos_normal"]).sum(axis=1)
    k = np.argmax(np.where(interior, un, -np.inf))      # nodo de la flecha maxima (centro de un paño)
    volumen = ((a + 2*t)*(b + 2*t) - a*b)*L
    return dict(
        nombre=nombre, t_mm=round(t*1e3), paso_mm=round(d["paso_s"]*1e3), carga=d["carga"],
        nodos=d["n_nodos"], gdl_resuelto=d["n_gdl_resuelto"], pernos=len(traccion),
        peso_kg=round(7850*volumen), flecha_mm=un[interior].max()*1e3,
        vm_lejos_MPa=vm[lejos].max()*1e-6, vm_centro_MPa=vm[k]*1e-6, fs_fluencia=fy/vm[lejos].max(),
        perno_max_kN=traccion.max()*1e-3, perno_prom_kN=traccion.mean()*1e-3,
        util_plancha=vm[lejos].max()/admisible_plancha, util_perno=traccion.max()/admisible_perno,
        seg=d["tiempos"]["solucion"],
    )


if __name__ == "__main__":
    carpeta.mkdir(exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        for nombre, estado in pool.map(lambda kv: resolver(*kv), casos.items()):
            print(f"{nombre:22s} {estado}", flush=True)

    filas = [metricas(n) for n in casos if (carpeta / f"{n}.npz").exists()]
    with open(carpeta / "variaciones.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in fila.items()} for fila in filas)

    print(f"\n{'t':>3} {'paso':>4} {'carga':>10} {'pernos':>6} {'peso kg':>8} {'flecha mm':>9} {'VM MPa':>7} "
          f"{'VM centro':>9} {'perno kN':>8} {'plancha':>8} {'perno':>6}")
    for q in filas:
        print(f"{q['t_mm']:3d} {q['paso_mm']:4d} {q['carga']:>10} {q['pernos']:6d} {q['peso_kg']:8d} "
              f"{q['flecha_mm']:9.3f} {q['vm_lejos_MPa']:7.1f} {q['vm_centro_MPa']:9.1f} {q['perno_max_kN']:8.1f} "
              f"{q['util_plancha']:8.0%} {q['util_perno']:6.0%}")
