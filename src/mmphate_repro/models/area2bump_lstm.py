from __future__ import annotations

from dataclasses import dataclass
import tensorflow as tf


@dataclass(frozen=True)
class LSTMConfig:
    n_units: int = 20
    n_classes: int = 8
    lr: float = 1e-4
    l2: float = 0.0
    dropout: float = 0.0
    recurrent_dropout: float = 0.0
    seed: int = 42


def build_lstm_classifier(input_shape: tuple[int, int], cfg: LSTMConfig):
    """
    Returns:
      model: classifier model
      trace_model: model that outputs the LSTM hidden sequence (for TraceHistory)
    """
    tf.keras.utils.set_random_seed(cfg.seed)

    inputs = tf.keras.Input(shape=input_shape)  # (T, F)

    reg = tf.keras.regularizers.l2(cfg.l2) if cfg.l2 and cfg.l2 > 0 else None
    lstm_layer = tf.keras.layers.LSTM(
        cfg.n_units,
        return_sequences=True,
        kernel_regularizer=reg,
        recurrent_regularizer=reg,
        dropout=cfg.dropout,
        recurrent_dropout=cfg.recurrent_dropout,
        name="lstm",
    )

    h_seq = lstm_layer(inputs)  # (batch, T, n_units)
    flat = tf.keras.layers.Flatten()(h_seq)
    outputs = tf.keras.layers.Dense(cfg.n_classes, activation="softmax")(flat)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # trace model outputs the LSTM sequence only
    trace_model = tf.keras.Model(inputs=inputs, outputs=h_seq)

    return model, trace_model