from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class HARLSTMConfig:
    n_units: int = 30
    n_classes: int = 6
    lr: float = 2e-5
    seed: int = 42


def build_har_lstm(input_shape: tuple[int, int], cfg: HARLSTMConfig):
    """
    Sequence-to-sequence LSTM classifier for HAR.

    Input shape: (T, F) = (128, 9)
    The model predicts the activity class at every timestep.
    Since HAR labels are constant per sequence, this is equivalent
    to sequence classification.

    Returns:
        model:       compiled classifier  (input -> softmax over n_classes per timestep)
        trace_model: returns LSTM hidden sequence  (input -> (T, n_units))
    """
    tf.keras.utils.set_random_seed(cfg.seed)

    inputs = tf.keras.Input(shape=input_shape)
    lstm_layer = tf.keras.layers.LSTM(
        cfg.n_units,
        return_sequences=True,
        name="lstm",
    )
    h_seq = lstm_layer(inputs)  # (batch, T, n_units)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(cfg.n_classes, activation="softmax"),
        name="output",
    )(h_seq)  # (batch, T, n_classes)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
        metrics=["accuracy"],
    )

    trace_model = tf.keras.Model(inputs=inputs, outputs=h_seq)
    return model, trace_model
