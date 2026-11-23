import matplotlib.pyplot as plt


COLORES = {"H8": "#c01c28", "H20": "#1a5fb4", "H27": "#26a269"}
LINEAS = ["-", "--", ":", "-."]
MARCAS = ["o", "s", "^", "D"]


## Curvas resultado / teorico. Un panel por cantidad comparada.
## series: lista de dict(label, tipo, x, ys=[y_panel_0, y_panel_1, ...], estilo=0, 1, ...)
##   tipo fija el color (H8 rojo, H20 azul, H27 verde) y estilo la linea y la marca
## ylabels: nombre de cada panel; referencia: linea horizontal (1 = igual al teorico)
def graficar_curvas(series, xlabel, ylabels, titulo, logx=False, referencia=1.0):
    fig, axs = plt.subplots(1, len(ylabels), figsize=(6*len(ylabels), 4.8), layout="constrained",
                            squeeze=False)
    for i, (ax, ylabel) in enumerate(zip(axs[0], ylabels)):
        for s in series:
            e = s.get("estilo", 0)
            ax.plot(s["x"], s["ys"][i], color=COLORES.get(s["tipo"], "k"), ls=LINEAS[e % 4],
                    marker=MARCAS[e % 4], ms=5, lw=1.4, label=s["label"])
        if referencia is not None:
            ax.axhline(referencia, color="0.3", lw=0.8, ls="--")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axs[0, -1].legend(fontsize=8, loc="best")
    fig.suptitle(titulo)
    return fig, axs[0]
