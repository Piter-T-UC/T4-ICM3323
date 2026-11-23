import numpy as np
import matplotlib.pyplot as plt

from etapa1_malla.malla_tunel import PAREDES


## Desarrollo (cara exterior desenrollada) del blindaje: una vista por pared con la malla,
## las uniones de las planchas y los pernos. Es la vista para disenar la distribucion.
## info: el dict que entrega malla_tunel (en metros); se dibuja en mm
def graficar_desarrollo(info, titulo="Desarrollo de la cara exterior"):
    mm = 1e3
    s_esq, zu = info["s_esquinas"], info["z_uniones"]
    s_lin, z_lin = info["s_lin"], info["z_lin"]
    pernos = info["pernos_sz"]
    largo = np.diff(s_esq)

    fig, axs = plt.subplots(2, 2, figsize=(16, 2*16*info["L"]/(largo[0] + largo[1]) + 2.5),
                            gridspec_kw={"width_ratios": largo[:2]}, layout="constrained", squeeze=False)
    for w, ax in zip([0, 1, 2, 3], axs.ravel()):
        s0, s1 = s_esq[w], s_esq[w + 1]
        en_pared = (pernos[:, 0] >= s0 - 1e-9) & (pernos[:, 0] <= s1 + 1e-9)
        for s in s_lin[(s_lin >= s0 - 1e-9) & (s_lin <= s1 + 1e-9)]:
            ax.axvline((s - s0)*mm, color="0.8", lw=0.4)
        for z in z_lin:
            ax.axhline(z*mm, color="0.8", lw=0.4)
        for z in zu:
            ax.axhline(z*mm, color="#1a5fb4", lw=2)
        ax.plot((pernos[en_pared, 0] - s0)*mm, pernos[en_pared, 1]*mm, "o", color="#c01c28", ms=4)
        ax.set_xlim(0, (s1 - s0)*mm)
        ax.set_ylim(0, info["L"]*mm)
        ax.set_aspect("equal")
        ax.set_xlabel("s a lo largo de la pared [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_title(f"Pared {PAREDES[w]} ({(s1 - s0)*mm:.0f} mm): {en_pared.sum()} pernos", fontsize=10)

    n_planchas = len(zu) - 1
    fig.suptitle(f"{titulo}\n{len(pernos)} pernos, {n_planchas} plancha(s) de {info['ancho_plancha']*mm:.0f} mm "
                 f"en z por pared (azul: uniones), {len(s_lin) - 1} x {len(z_lin) - 1} elementos en s x z")
    return fig, axs
