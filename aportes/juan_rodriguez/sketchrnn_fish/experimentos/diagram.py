"""Dibuja el diagrama de la arquitectura final Conv-LSTM (entradas, procesamiento y salidas en entrenamiento).

Uso: python diagram.py salida.png [conv_channels conv_layers kernel hidden layers dropout n_mix w_pen]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C_IN, C_CONV, C_RNN, C_REG, C_PEN, C_LOSS, C_AUX = ("#dbeafe", "#e0f2fe", "#ede9fe", "#dcfce7", "#fef3c7",
                                                    "#fee2e2", "#f1f5f9")
EDGE, INK, ACCENT = "#334155", "#0f172a", "#7c3aed"


def box(ax, x, y, w, h, text, color, fs=10.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=EDGE, lw=1.3, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK, linespacing=1.45)


def arrow(ax, p, q, color=EDGE, ls="-", rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def label(ax, x, y, text, fs=9, color=EDGE, ha="left"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va="center")


def draw(path, conv_channels=128, conv_layers=2, kernel=3, hidden=512, layers=1, dropout=0.1, n_mix=20,
         w_pen=3.0, w_mse=1.0):
    receptive = conv_layers * (kernel - 1) + 1
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.text(9, 11.65, "Arquitectura del modelo: Conv-LSTM autorregresivo + mezcla de Gaussianas para (Δx, Δy) "
                      "+ clasificador de pen_state", ha="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(9, 11.25, "Flujo de datos en entrenamiento (teacher forcing): en cada paso t el modelo ve los puntos "
                      "reales anteriores y predice el punto t", ha="center", fontsize=10.5, color=EDGE)

    # ---------------- flujo principal (columna central) ----------------
    X0, W = 2.6, 7.8
    cx = X0 + W / 2
    box(ax, X0, 9.55, W, 1.4,
        "Entrada: lote $X$ de forma $(B,\\ T \\leq 200,\\ 5)$\n"
        "$x_t = [\\Delta x_{t-1}/\\sigma_x,\\ \\Delta y_{t-1}/\\sigma_y,\\ p_1,\\ p_2,\\ p_3]$   (stroke-5 normalizado)\n"
        "$x_1 = S_0 = [0, 0, 1, 0, 0]$ (punto inicial / semilla)", C_IN, fs=11)
    arrow(ax, (cx, 9.55), (cx, 9.15))
    box(ax, X0, 7.75, W, 1.4,
        f"Frente convolucional causal: {conv_layers} × [Conv1d(k = {kernel}, {conv_channels} canales) + GELU]\n"
        "relleno solo a la izquierda: la salida en $t$ solo usa $x_{\\leq t}$\n"
        f"rasgos locales de los últimos {receptive} puntos → forma $(B,\\ T,\\ {conv_channels})$", C_CONV, fs=11)
    arrow(ax, (cx, 7.75), (cx, 7.35))
    layer_txt = f"{layers} capa" + ("s" if layers > 1 else "")
    box(ax, X0, 5.85, W, 1.5,
        f"LSTM ({layer_txt}, {hidden} unidades ocultas)\n"
        "$h_t,\\ c_t = \\mathrm{LSTM}(\\mathrm{conv}_t,\\ h_{t-1},\\ c_{t-1})$\n"
        f"salida $H$ de forma $(B,\\ T,\\ {hidden})$: $h_t$ resume TODO el contexto previo\n"
        "(contexto $c = t-1$ puntos, sin ventana fija)", C_RNN, fs=11)
    arrow(ax, (cx, 5.85), (cx, 5.45))
    box(ax, X0, 4.9, W, 0.55, f"Dropout (p = {dropout})", C_RNN, fs=11)
    xr, xp = X0 + 1.85, X0 + W - 1.85                     # centros de las dos ramas
    arrow(ax, (xr, 4.9), (xr, 4.62))
    arrow(ax, (xp, 4.9), (xp, 4.62))
    box(ax, X0, 2.82, 3.7, 1.8,
        "Cabeza de regresión\n"
        f"Linear({hidden} → 6·{n_mix} = {6 * n_mix})\n"
        "$\\pi_k$ (softmax),  $\\mu_{x,k},\\ \\mu_{y,k}$\n"
        "$\\sigma_{x,k},\\ \\sigma_{y,k}$ (softplus),  $\\rho_k$ (tanh)\n"
        "Salidas: $\\Delta\\hat{x}_t,\\ \\Delta\\hat{y}_t = \\Sigma_k\\, \\pi_k \\mu_k$", C_REG, fs=10)
    box(ax, X0 + W - 3.7, 2.82, 3.7, 1.8,
        "Cabeza de clasificación\n"
        f"Linear({hidden} → 3)\n"
        "logits de pen_state\n(lápiz abajo / fin trazo / fin dibujo)\n"
        "Salida: $\\widehat{pen}_t$ (clase más probable)", C_PEN, fs=10)
    arrow(ax, (xr, 2.82), (xr, 2.55))
    arrow(ax, (xp, 2.82), (xp, 2.55))
    box(ax, X0, 1.4, 3.7, 1.15,
        "Pérdidas de regresión\n"
        "$\\mathcal{L}_{NLL} = -\\log \\Sigma_k\\, \\pi_k\\, \\mathcal{N}(\\Delta x_t, \\Delta y_t \\mid \\mu_k, \\sigma_k, \\rho_k)$\n"
        "$\\mathcal{L}_{MSE} = \\frac{1}{2}\\,\\|\\Sigma_k\\, \\pi_k \\mu_k - (\\Delta x_t, \\Delta y_t)\\|^2$",
        C_LOSS, fs=9.6)
    box(ax, X0 + W - 3.7, 1.4, 3.7, 1.15,
        "Pérdida de clasificación (CE ponderada)\n"
        "$\\mathcal{L}_{pen} = w_c \\cdot \\mathrm{CE}(\\mathrm{logits}_t,\\ pen_t)$\n"
        "$w_c \\propto 1/\\sqrt{\\mathrm{frecuencia}_c}$",
        C_LOSS, fs=9.6)
    arrow(ax, (xr, 1.4), (cx - 0.6, 0.95))
    arrow(ax, (xp, 1.4), (cx + 0.6, 0.95))
    box(ax, X0 + 0.5, 0.2, W - 1.0, 0.75,
        "$\\mathcal{L} = \\Sigma_{b,t}\\, m_{b,t}\\,(\\mathcal{L}_{NLL} + " + f"{w_mse:g}" + "\\cdot\\mathcal{L}_{MSE} + "
        + f"{w_pen:g}" + "\\cdot\\mathcal{L}_{pen}) \\;/\\; \\Sigma_{b,t}\\, m_{b,t}$", C_LOSS, fs=12, bold=True)

    # objetivos (entrada desplazada un paso) y mascara
    for bx, txt in [(0.15, "Objetivo\n$(\\Delta x_t,\\ \\Delta y_t)$\n(normalizado)"),
                    (X0 + W + 0.3, "Objetivo\n$pen_t \\in \\{0, 1, 2\\}$\n(clase)")]:
        box(ax, bx, 1.4, 2.15, 1.15, txt, C_IN, fs=10)
        box(ax, bx, 2.82, 2.15, 1.8, "$Y$ = punto $t$\n(la entrada $X$\ndesplazada\nun paso)", C_IN, fs=10)
        arrow(ax, (bx + 1.075, 2.82), (bx + 1.075, 2.55))
    arrow(ax, (2.3, 1.97), (X0, 1.97))
    arrow(ax, (X0 + W + 0.3, 1.97), (X0 + W, 1.97))
    box(ax, X0 + W + 0.3, 0.02, 2.15, 1.15,
        "Máscara $m_{b,t}$\n1: punto real (trazos\n+ 1 punto 'fin')\n0: relleno", C_AUX, fs=9.2, ls="--")
    arrow(ax, (X0 + W + 0.3, 0.58), (X0 + W - 0.5, 0.58))

    # ---------------- columna derecha: desenrollado y generacion ----------------
    RX = 13.2
    ax.text(RX + 2.3, 10.75, "Desenrollado en el tiempo", ha="center", fontsize=11.5, fontweight="bold", color=INK)
    xs = [RX + 0.6, RX + 2.3, RX + 4.0]
    for i, (x, li, lo) in enumerate(zip(xs, ["$S_0$", "$P_1$", "$P_{t-1}$"],
                                         ["$\\hat{P}_1$", "$\\hat{P}_2$", "$\\hat{P}_t$"])):
        box(ax, x - 0.5, 6.75, 1.0, 0.5, li, C_IN, fs=11)
        box(ax, x - 0.62, 7.6, 1.24, 0.55, "Conv", C_CONV, fs=10, bold=True)
        box(ax, x - 0.62, 8.55, 1.24, 0.65, "LSTM", C_RNN, fs=10.5, bold=True)
        box(ax, x - 0.5, 9.65, 1.0, 0.5, lo, C_REG, fs=11)
        arrow(ax, (x, 7.25), (x, 7.6))
        arrow(ax, (x, 8.15), (x, 8.55))
        arrow(ax, (x, 9.2), (x, 9.65))
        if i < 2:
            arrow(ax, (x + 0.62, 8.87), (xs[i + 1] - 0.62, 8.87))
            arrow(ax, (x + 0.5, 7.0), (xs[i + 1] - 0.62, 7.75), color="#64748b", lw=1.0)   # ventana local
    label(ax, RX + 1.45, 9.12, "$h, c$", fs=9.5, ha="center")
    label(ax, RX + 3.15, 9.12, "···", fs=12, ha="center")
    for i in range(2):                                   # en generacion: salida muestreada -> siguiente entrada
        arrow(ax, (xs[i] + 0.5, 9.9), (xs[i + 1] - 0.5, 7.05), color=ACCENT, ls="--", rad=-0.3, lw=1.2)
    label(ax, RX + 2.3, 6.35, "- - -  en generación: $\\hat{P}_t$ muestreado se realimenta como $x_{t+1}$",
          fs=9, color=ACCENT, ha="center")

    ax.text(RX + 2.3, 5.75, "Generación (inferencia)", ha="center", fontsize=11.5, fontweight="bold", color=INK)
    box(ax, RX - 0.1, 3.4, 4.8, 2.2,
        "1. $x_1 = S_0$ (semilla); estado = (h, c, últimas entradas)\n"
        "2. muestrear $(\\Delta x, \\Delta y)$ de la mezcla con\n    temperatura $\\tau$, y pen_state de\n"
        "    softmax(logits $- \\log w$) (corrección de priors)\n"
        "3. el punto generado es la entrada $x_{t+1}$\n"
        "4. parar si pen_state = fin o $t = 200$;\n    des-normalizar y renderizar", C_AUX, fs=9.8)

    ax.text(RX + 2.3, 2.95, "Métricas en prueba", ha="center", fontsize=11.5, fontweight="bold", color=INK)
    box(ax, RX - 0.1, 0.4, 4.8, 2.25,
        "Predicción a 1 paso, por punto\n(todos los puntos reales juntos):\n"
        "• MSE$(\\Delta x, \\Delta y)$ con la media\n   de la mezcla $\\Sigma_k\\, \\pi_k \\mu_k$\n"
        "• F1 macro de pen_state (3 clases)\n• Pérdida $\\mathcal{L}$ (misma máscara)", C_AUX, fs=9.8)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) > 1:
        draw(a[0], int(a[1]), int(a[2]), int(a[3]), int(a[4]), int(a[5]), float(a[6]), int(a[7]), float(a[8]),
             float(a[9]))
    else:
        draw(a[0])
