import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIGURATION
# ============================================================

LATENT_DIM = 24
BETA = 0.001
EPOCHS = 30
BATCH_SIZE = 512
LEARNING_RATE = 0.001

DATA_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

X_train = np.load(f"{DATA_DIR}/X_train.npy").astype("float32")
X_val = np.load(f"{DATA_DIR}/X_val.npy").astype("float32")

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)


# ============================================================
# ENCODER
# 39 → 64 → 32 → 24
# ============================================================

encoder_inputs = keras.Input(
    shape=(X_train.shape[1],),
    name="encoder_input"
)

x = layers.Dense(64, activation="relu")(encoder_inputs)
x = layers.Dense(32, activation="relu")(x)

z_mean = layers.Dense(
    LATENT_DIM,
    name="z_mean"
)(x)

z_log_var = layers.Dense(
    LATENT_DIM,
    name="z_log_var"
)(x)


# ------------------------------------------------------------
# Sampling layer
# ------------------------------------------------------------

class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs

        epsilon = tf.random.normal(
            shape=tf.shape(z_mean)
        )

        return z_mean + tf.exp(
            0.5 * z_log_var
        ) * epsilon


z = Sampling(name="z")([z_mean, z_log_var])


encoder = keras.Model(
    encoder_inputs,
    [z_mean, z_log_var, z],
    name="vae_encoder"
)

encoder.summary()


# ============================================================
# DECODER
# 24 → 32 → 64 → 39
# ============================================================

latent_inputs = keras.Input(
    shape=(LATENT_DIM,),
    name="decoder_input"
)

x = layers.Dense(32, activation="relu")(latent_inputs)
x = layers.Dense(64, activation="relu")(x)

decoder_outputs = layers.Dense(
    X_train.shape[1],
    activation="linear",
    name="decoder_output"
)(x)

decoder = keras.Model(
    latent_inputs,
    decoder_outputs,
    name="vae_decoder"
)

decoder.summary()


# ============================================================
# VAE MODEL
# ============================================================

class VAE(keras.Model):

    def __init__(
        self,
        encoder,
        decoder,
        beta=0.001,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

        self.total_loss_tracker = keras.metrics.Mean(
            name="loss"
        )

        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

        self.kl_loss_tracker = keras.metrics.Mean(
            name="kl_loss"
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):

        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:

            # ------------------------------------------------
            # Encode
            # ------------------------------------------------

            z_mean, z_log_var, z = self.encoder(
                data,
                training=True
            )

            # ------------------------------------------------
            # Decode
            # ------------------------------------------------

            reconstruction = self.decoder(
                z,
                training=True
            )

            # ------------------------------------------------
            # Reconstruction loss
            # ------------------------------------------------

            reconstruction_loss = tf.reduce_mean(
                tf.square(
                    data - reconstruction
                )
            )

            # ------------------------------------------------
            # KL divergence
            # ------------------------------------------------

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1
                    + z_log_var
                    - tf.square(z_mean)
                    - tf.exp(z_log_var),
                    axis=1
                )
            )

            # ------------------------------------------------
            # Total β-VAE loss
            # ------------------------------------------------

            total_loss = (
                reconstruction_loss
                + self.beta * kl_loss
            )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        gradients = tape.gradient(
            total_loss,
            self.trainable_weights
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.trainable_weights
            )
        )

        # ----------------------------------------------------
        # Update metrics
        # ----------------------------------------------------

        self.total_loss_tracker.update_state(
            total_loss
        )

        self.reconstruction_loss_tracker.update_state(
            reconstruction_loss
        )

        self.kl_loss_tracker.update_state(
            kl_loss
        )

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss":
                self.reconstruction_loss_tracker.result(),
            "kl_loss":
                self.kl_loss_tracker.result(),
        }

    def test_step(self, data):

        if isinstance(data, tuple):
            data = data[0]

        # ----------------------------------------------------
        # IMPORTANT:
        # Use z_mean instead of sampled z for validation.
        # This makes validation deterministic.
        # ----------------------------------------------------

        z_mean, z_log_var, _ = self.encoder(
            data,
            training=False
        )

        reconstruction = self.decoder(
            z_mean,
            training=False
        )

        # ----------------------------------------------------
        # Reconstruction loss
        # ----------------------------------------------------

        reconstruction_loss = tf.reduce_mean(
            tf.square(
                data - reconstruction
            )
        )

        # ----------------------------------------------------
        # KL divergence
        # ----------------------------------------------------

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1
                + z_log_var
                - tf.square(z_mean)
                - tf.exp(z_log_var),
                axis=1
            )
        )

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        total_loss = (
            reconstruction_loss
            + self.beta * kl_loss
        )

        self.total_loss_tracker.update_state(
            total_loss
        )

        self.reconstruction_loss_tracker.update_state(
            reconstruction_loss
        )

        self.kl_loss_tracker.update_state(
            kl_loss
        )

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss":
                self.reconstruction_loss_tracker.result(),
            "kl_loss":
                self.kl_loss_tracker.result(),
        }


# ============================================================
# CREATE VAE
# ============================================================

vae = VAE(
    encoder=encoder,
    decoder=decoder,
    beta=BETA,
    name="beta_vae_24"
)

vae.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )
)


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING β-VAE TRAINING")
print("=" * 60)

print(f"Latent dimension : {LATENT_DIM}")
print(f"Beta             : {BETA}")
print(f"Batch size       : {BATCH_SIZE}")
print(f"Epochs            : {EPOCHS}")
print("=" * 60 + "\n")


history = vae.fit(
    X_train,
    X_train,
    validation_data=(
        X_val,
        X_val
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# SAVE ENCODER + DECODER
# ============================================================

encoder_path = (
    f"{MODEL_DIR}/vae_beta_encoder_24.keras"
)

decoder_path = (
    f"{MODEL_DIR}/vae_beta_decoder_24.keras"
)

encoder.save(encoder_path)
decoder.save(decoder_path)


# ============================================================
# PRINT RESULTS
# ============================================================

best_epoch = np.argmin(
    history.history["val_loss"]
) + 1

best_val_loss = min(
    history.history["val_loss"]
)

best_val_reconstruction = (
    history.history["val_reconstruction_loss"][
        best_epoch - 1
    ]
)

best_val_kl = (
    history.history["val_kl_loss"][
        best_epoch - 1
    ]
)

print("\n" + "=" * 60)
print("β-VAE TRAINING COMPLETED")
print("=" * 60)

print(f"Best epoch              : {best_epoch}")
print(f"Best validation loss    : {best_val_loss:.6f}")
print(
    f"Validation reconstruction: "
    f"{best_val_reconstruction:.6f}"
)
print(
    f"Validation KL loss      : "
    f"{best_val_kl:.6f}"
)

print("\nModels saved:")
print(f"Encoder: {encoder_path}")
print(f"Decoder: {decoder_path}")

print("=" * 60)