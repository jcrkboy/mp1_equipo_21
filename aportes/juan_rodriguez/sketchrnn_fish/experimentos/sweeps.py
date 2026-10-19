"""Definicion de los barridos de experimentos (cada funcion devuelve una lista de configuraciones).

Estrategia (successive halving): la ronda 1 hace un "screening" corto y comparable de muchas
familias (mismo presupuesto de epocas, LR coseno); las rondas siguientes afinan solo lo mejor.
"""

SCREEN = dict(epochs=15, sched="cosine", patience=100)


def smoke():
    q = dict(n_train=800, n_valid=200, n_test=200, epochs=2, gen_n=10)
    return [dict(q, name=f"smoke_{m}_{h}", model=m, head=h)
            for m, h in [("lstm", "gmm"), ("gru", "mse"), ("rnn", "gauss"),
                         ("transformer", "gmm"), ("tcn", "gmm"), ("mlp", "mse")]]


def r1():
    """Ronda 1: familias de arquitectura (con cabeza GMM) y tipo de cabeza de regresion (con LSTM)."""
    cfgs = [
        dict(name="r1_lstm512_gmm", model="lstm", hidden=512),
        dict(name="r1_lstm256x2_gmm", model="lstm", hidden=256, layers=2, dropout=0.2),
        dict(name="r1_trf128x4_gmm", model="transformer", d_model=128, nhead=4, layers=4, ff=512, lr=5e-4),
        dict(name="r1_lstm256_gmm", model="lstm", hidden=256),
        dict(name="r1_gru256_gmm", model="gru", hidden=256),
        dict(name="r1_lstm256_mse", model="lstm", hidden=256, head="mse"),
        dict(name="r1_lstm256_gauss", model="lstm", hidden=256, head="gauss"),
        dict(name="r1_gru256_mse", model="gru", hidden=256, head="mse"),
        dict(name="r1_rnn256_gmm", model="rnn", hidden=256),
        dict(name="r1_tcn128_gmm", model="tcn", channels=128, levels=5, kernel=3),
        dict(name="r1_lstm128_gmm", model="lstm", hidden=128),
        dict(name="r1_mlp16_gmm", model="mlp", context=16, hidden=256),
        dict(name="r1_mlp16_mse", model="mlp", context=16, hidden=256, head="mse"),
    ]
    return [dict(SCREEN, **c) for c in cfgs]


def r2():
    """Ronda 2.

    (a) Refinamiento de arquitectura, motivado por la ronda 1 (la TCN supero al LSTM): LSTM con capa de
        integracion (posicion absoluta) y/o frente convolucional causal (Conv-LSTM); TCN con distinta
        capacidad y campo receptivo. (r2_lstm256_pos y r2_lstm256_conv ya corrieron.)
    (b) Perdida y optimizacion, un factor a la vez sobre Conv-LSTM-256 (la base recurrente ganadora).
        Los regularizadores (dropout, aumento de datos) se evaluan en la ronda 3, con entrenamiento largo.
    (c) Estudio del tamano de contexto c con el MLP de ventana fija.
    """
    arch = [
        dict(name="r2_lstm512_conv", model="lstm", hidden=512, conv_front=128),
        dict(name="r2_lstm256_pos", model="lstm", hidden=256, abs_pos=True),
        dict(name="r2_lstm256_conv", model="lstm", hidden=256, conv_front=128),
        dict(name="r2_lstm256_posconv", model="lstm", hidden=256, abs_pos=True, conv_front=128),
        dict(name="r2_lstm256_conv3k5", model="lstm", hidden=256, conv_front=128, conv_layers=3, conv_kernel=5),
        dict(name="r2_lstm256_conv256", model="lstm", hidden=256, conv_front=256),
        dict(name="r2_gru256_conv", model="gru", hidden=256, conv_front=128),
        dict(name="r2_tcn128_l6", model="tcn", channels=128, levels=6, kernel=3),
        dict(name="r2_tcn128_k2", model="tcn", channels=128, levels=5, kernel=2),
        dict(name="r2_tcn128_l3", model="tcn", channels=128, levels=3, kernel=3),
    ]
    tcn = [c for c in arch if c["model"] == "tcn"]          # las TCN, al final (menor prioridad)
    cfgs = [dict(SCREEN, **c) for c in arch if c["model"] != "tcn"]
    base = dict(SCREEN, model="lstm", hidden=256, conv_front=128)
    variants = [
        ("cw_sqrt", dict(class_weights="sqrt")),
        ("cw_inv", dict(class_weights="inv")),
        ("wpen3", dict(w_pen=3.0)),
        ("mseaux1", dict(mse_aux=1.0)),
        ("combo", dict(mse_aux=1.0, w_pen=3.0, class_weights="sqrt")),
        ("mix10", dict(n_mix=10)),
        ("lr2e3", dict(lr=2e-3)),
        ("bs64", dict(batch_size=64)),
    ]
    cfgs += [dict(base, name=f"r2_clstm256_{k}", **v) for k, v in variants]
    cfgs += [dict(SCREEN, **c) for c in tcn]
    # (c) estudio del tamano de contexto c con el modelo de ventana fija (barato): c = 4, 8, 32 (c = 16 en r1)
    cfgs += [dict(SCREEN, name=f"r2_mlp{c}_gmm", model="mlp", context=c, hidden=256) for c in (4, 8, 32)]
    return cfgs


# entrenamiento largo, igual al del notebook: LR con ReduceLROnPlateau + early stopping sobre la perdida de val.
LONG = dict(epochs=60, sched="plateau", sched_patience=3, patience=8)
LOSS = dict(mse_aux=1.0, w_pen=3.0, class_weights="sqrt")          # la mejor combinacion de la ronda 2b
FINAL = dict(LONG, **LOSS, lr=2e-3, model="lstm", hidden=512, conv_front=128, conv_layers=2, conv_kernel=3)


def r3():
    """Ronda 3: candidatos finales (Conv-LSTM-512) con entrenamiento largo; regularizacion y LR."""
    return [
        dict(FINAL, name="r3_clstm512_d01"),
        dict(FINAL, name="r3_clstm512_d03", dropout=0.3),
        dict(FINAL, name="r3_clstm512_aug", augment=0.15),
        dict(FINAL, name="r3_clstm512_lr1e3", lr=1e-3),
    ]


def r3tcn():
    """Referencia de la ronda 3: la TCN con la misma perdida y el mismo entrenamiento largo."""
    return [dict(LONG, **LOSS, lr=2e-3, name="r3_tcn128_d01", model="tcn", channels=128, levels=5, kernel=3)]
