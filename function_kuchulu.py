import tensorflow as tf

def my_callback(filepath):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='min',
        save_weights_only=False
    )
(*     early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=1,
        mode='min' *)

    return [checkpoint]
