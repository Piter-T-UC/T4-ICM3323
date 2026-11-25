import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# Figuras del barrido de variaciones.py (lee resultados/variaciones.csv):
#   variaciones_espesor_paso.png  flecha, von Mises y fuerza en pernos vs espesor, una linea por paso
#   variaciones_utilizacion.png   mapas espesor x paso: plancha (VM / 235 MPa) y perno (T / 69.3 kN)
#   python3 graficar_variaciones.py

carpeta = Path(__file__).parent / "resultados"
with open(carpeta / "variaciones.csv", encoding="utf-8") as f:
    filas = [{k: (v if k in ("nombre", "carga") else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
hidro = [r for r in filas if r["carga"] == "hidrostatica"]
espesores = sorted({r["t_mm"] for r in hidro})
pasos = sorted({r["paso_mm"] for r in hidro})

# Paleta (rampa secuencial azul para el paso, que es una magnitud ordenada: mas paso = mas oscuro)
superficie, tinta, tinta2, grilla = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
azul = {250: "#86b6ef", 400: "#3987e5", 500: "#256abf", 650: "#104281"}
color_paso = dict(zip(pasos, (azul[250], azul[400], azul[500], azul[650])))
rojo = "#d03b3b"        # estado "critico": solo para el limite admisible
plt.rcParams.update({"font.size": 9, "axes.edgecolor": grilla, "axes.labelcolor": tinta2,
                     "xtick.color": tinta2, "ytick.color": tinta2, "text.color": tinta,
                     "figure.facecolor": superficie, "axes.facecolor": superficie})


def serie(paso, clave):
    q = sorted((r for r in hidro if r["paso_mm"] == paso), key=lambda r: r["t_mm"])
    return [r["t_mm"] for r in q], [r[clave] for r in q]


# --- 1. Lineas vs espesor ------------------------------------------------------------------------
paneles = [("flecha_mm", "Flecha maxima hacia adentro [mm]", True, None),
           ("vm_lejos_MPa", "von Mises maximo a > 100 mm de pernos [MPa]", False, (235, "admisible plancha 235 MPa")),
           ("perno_max_kN", "Traccion maxima en un perno [kN]", False, (69.3, "admisible perno 69.3 kN"))]
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
for ax, (clave, titulo, log, limite) in zip(axs, paneles):
    for p in pasos:
        x, y = serie(p, clave)
        ax.plot(x, y, color=color_paso[p], lw=2, marker="o", ms=5, mec=superficie, mew=1.5, label=f"paso {p:.0f} mm")
    if limite:
        ax.axhline(limite[0], color=rojo, lw=1, ls="--")
        ax.text(espesores[-1], limite[0], limite[1] + " ", color=tinta2, ha="right", va="bottom", fontsize=8)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(espesores)
    ax.set_xlabel("espesor de plancha t [mm]")
    ax.set_title(titulo, loc="left", fontsize=10, color=tinta)
    ax.grid(axis="y", color=grilla, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(18, color=tinta2, lw=0.8, ls=":")
axs[0].text(18.2, axs[0].get_ylim()[1], "t de calculo\n(18 mm)", color=tinta2, fontsize=8, va="top")
axs[0].legend(frameon=False, loc="lower left", fontsize=8)
fig.suptitle("Blindaje del tunel (H27, h = 50 mm, cuarto de modelo, perno = cara de un hexaedro de 50 mm, "
             "presion hidrostatica): efecto del espesor y del paso de pernos", x=0.01, ha="left", fontsize=11)
fig.savefig(carpeta / "variaciones_espesor_paso.png", dpi=150)


# --- 2. Mapas de utilizacion ----------------------------------------------------------------------
ramp = LinearSegmentedColormap.from_list("azul", ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#104281", "#0d366b"])
fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, clave, titulo in zip(axs, ("util_plancha", "util_perno"),
                             ("Plancha: von Mises / 235 MPa", "Perno: traccion / 69.3 kN")):
    M = np.array([[next(r[clave] for r in hidro if r["t_mm"] == t and r["paso_mm"] == p) for p in pasos]
                  for t in espesores])
    ax.imshow(M, cmap=ramp, vmin=0, vmax=max(1.5, M.max()), aspect="auto", origin="lower")
    for i in range(len(espesores)):
        for j in range(len(pasos)):
            v = M[i, j]
            ax.text(j, i, f"{v:.0%}" + (" !" if v > 1 else ""), ha="center", va="center", fontsize=9,
                    color="white" if v > 0.6*max(1.5, M.max()) else tinta, fontweight="bold" if v > 1 else "normal")
    ax.set_xticks(range(len(pasos)), [f"{p:.0f}" for p in pasos])
    ax.set_yticks(range(len(espesores)), [f"{t:.0f}" for t in espesores])
    ax.set_xlabel("paso de pernos [mm]")
    ax.set_ylabel("espesor t [mm]")
    ax.set_title(titulo, loc="left", fontsize=10, color=tinta)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
fig.suptitle("Utilizacion respecto al admisible del informe (! = lo supera)", x=0.01, ha="left", fontsize=11)
fig.savefig(carpeta / "variaciones_utilizacion.png", dpi=150)
print("Figuras: variaciones_espesor_paso.png, variaciones_utilizacion.png en", carpeta)
