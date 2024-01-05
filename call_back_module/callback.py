# Callback to back up and restore the training state.
backup = tf.keras.callbacks.BackupAndRestore(
  backup_dir,
  save_freq='epoch',
  delete_checkpoint=True,
  save_before_preemption=False
)

'''
example:
callback = tf.keras.callbacks.BackupAndRestore(backup_dir="/tmp/backup")
'''

# Callback that accumulates epoch averages of metrics.
tf.keras.callbacks.BaseLogger(
    stateful_metrics=None
)
# Callback that streams epoch results to a CSV file.
tf.keras.callbacks.CSVLogger(
    filename, separator=',', append=False
)

'''
Example:
csv_logger = CSVLogger('training.log')
model.fit(X_train, Y_train, callbacks=[csv_logger])
'''
# Abstract base class used to build new callbacks.
tf.keras.callbacks.Callback()
'''
example:
class MyCallback(tf.keras.callbacks.Callback):
def on_train_end(self, logs=None):
  global training_finished
  training_finished = True
'''
# Container abstracting a list of callbacks.
tf.keras.callbacks.CallbackList(
  callbacks=None, add_history=False, add_progbar=False, model=None, **params
)

'''
  callbacks = keras.callbacks.CallbackList([
    ReduceLROnPlateau(monitor="loss", factor=0.1, patience=3),
    EarlyStopping(monitor="loss", min_delta=0.01, patience=5),
    PrintLR()
])
'''
# Stop training when a monitored metric has stopped improving.
tf.keras.callbacks.EarlyStopping(
  monitor='val_loss',
  min_delta=0,
  patience=0,
  verbose=0,
  mode='auto',
  baseline=None,
  restore_best_weights=False,
  start_from_epoch=0
)
# Callback that records events into a History object.
tf.keras.callbacks.History()
'''
example:
history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
                  epochs=10, verbose=1)
'''
# Callback for creating simple, custom callbacks on-the-fly.
tf.keras.callbacks.LambdaCallback(
    on_epoch_begin=None,
    on_epoch_end=None,
    on_batch_begin=None,
    on_batch_end=None,
    on_train_begin=None,
    on_train_end=None,
    **kwargs
)
'''
example:
# Stream the epoch loss to a file in JSON format. The file content
# is not well-formed JSON but rather has a JSON object per line.
import json
json_log = open('loss_log.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
  on_epoch_end=lambda epoch, logs: json_log.write(
      json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
  on_train_end=lambda logs: json_log.close()
)
'''
# Learning rate scheduler.

tf.keras.callbacks.LearningRateScheduler(
    schedule, verbose=0
)
'''
def scheduler(epoch, lr):
if epoch < 10:
  return lr
else:
  return lr * tf.math.exp(-0.1)
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
'''
# Callback to save the Keras model or model weights at some frequency.
tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor: str = 'val_loss',
    verbose: int = 0,
    save_best_only: bool = False,
    save_weights_only: bool = False,
    mode: str = 'auto',
    save_freq='epoch',
    options=None,
    initial_value_threshold=None,
    **kwargs
).
      
'''
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
filepath=checkpoint_filepath,
save_weights_only=True,
monitor='val_accuracy',
mode='max',
save_best_only=True)
model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])
'''

# Callback that prints metrics to stdout.
tf.keras.callbacks.ProgbarLogger(
  count_mode: str = 'samples', stateful_metrics=None
)

# Reduce learning rate when a metric has stopped improving.
tf.keras.callbacks.ReduceLROnPlateau(
  monitor='val_loss',
  factor=0.1,
  patience=10,
  verbose=0,
  mode='auto',
  min_delta=0.0001,
  cooldown=0,
  min_lr=0,
  **kwargs
)

# Callback used to stream events to a server.
tf.keras.callbacks.RemoteMonitor(
  root='http://localhost:9000',
  path='/publish/epoch/end/',
  field='data',
  headers=None,
  send_as_json=False
)

# Callback to save the best Keras model.
tf.keras.callbacks.SidecarEvaluatorModelExport(
  export_filepath, checkpoint_filepath, **kwargs
)

# Enable visualizations for TensorBoard.
tf.keras.callbacks.TensorBoard(
  log_dir='logs',
  histogram_freq=0,
  write_graph=True,
  write_images=False,
  write_steps_per_second=False,
  update_freq='epoch',
  profile_batch=0,
  embeddings_freq=0,
  embeddings_metadata=None,
  **kwargs
)

# Callback that terminates training when a NaN loss is encountered.
tf.keras.callbacks.TerminateOnNaN()

return []



