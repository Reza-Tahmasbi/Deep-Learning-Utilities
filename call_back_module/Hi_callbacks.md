## Callbacks are special utility functions that are executed during training at given stages of the training procedure.

They are levereged to prevent overfitting, visualize training progress, debug your code, save checkpoints, generate logs, create a TensorBoard, etc.
These funcitons are called when certain events are triggered. such as:

1.  ``on_epoch_end``: this is triggered when an epoch ends
2.  ``on_epoch_begin``: this is triggered when a new batch is passed for training.
3.  ``on_batch_end``: when a batch is finished with training.
4.  ``on_train_begin``: when the training starts.
5.  ``on_train_end``: when the training ends.

Usage: add any of them in your fit funciton.
```
model.fit(x, y, callbacks=list_of_callbacks)
```
Here are some practical callbacks in Tensorflow 2.0:

### 1. EarlyStopping
   
```
tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                min_delta=0, 
                                patience=0, 
                                verbose=0, 
                                mode='auto', 
                                baseline=None, 
                                restore_best_weights=False)
```
``Monitor:``
<details>
<summary> to monitor a specific metric</summary>
  ``loss`` The loss value of the training data.
  ``val_loss``: The loss value of the validation data.
  ``accuracy``: The accuracy of the training data.
  ``val_accuracy``: The accuracy of the validation data.
</details>

``min_delta``: the minimum amount of improvement we expect in every epoch.
``patience``: the number of epochs to wait before stopping the training.
``verbose``: whether or not to print additional logs. (0 , 1)
``mode``: defines whether the monitored metrics should be increasing, decreasing, or inferred from the name; possible values are 'min', 'max', or 'auto'.
``baseline``: values for the monitored metrics.
``restore_best_weights``: if set to True, the model will get the weights of the epoch which has the best value for the monitored metrics; otherwise, it will get the weights of the last epoch.

### 2. ModelCheckpoint
This callback allows us to save the model regularly during training. This is especially useful when training deep learning models which take a long time to train. This callback monitors the training and saves model checkpoints at regular intervals, based on the metrics.

```
tf.keras.callbacks.ModelCheckpoint(filepath, 
                                     monitor='val_loss', 
                                     verbose=0, 
                                     save_best_only=False,
                                     save_weights_only=False, 
                                     mode='auto', 
                                     save_freq='epoch')
```
``filepath``: path for saving the model. You can pass the file path with formatting options like model-{epoch:02d}-{val_loss:0.2f}; this saves the model with the mentioned values in the name.
``monitor``: name of the metrics to monitor.
save_best_only: if True, the best model will not be overridden.
``mode``: defines whether the monitored metrics should be increasing, decreasing, or inferred from the name; possible values are 'min', 'max', or 'auto'.
``save_weights_only``: if True, only the weights of the models will be saved. Otherwise the full model will be saved.
``save_freq``: if 'epoch', the model will be saved after every epoch. If an integer value is passed, the model will be saved after the integer number of batches (not to be confused with epochs).

The ModelCheckpoint callback is executed via the on_epoch_end trigger of training.


### 3. TensorBoard
This callback is a great choice for seeing the training summary of your model. It makes the logs for TensorBoard, which you can use to see how your training is going. We will explain more about TensorBoard in another article.
```
tf.keras.callbacks.TensorBoard(log_dir='logs',
                                 histogram_freq=0, 
                                 write_graph=True, 
                                 write_images=False,    
                                 update_freq='epoch', 
                                 profile_batch=2, 
                                 embeddings_freq=0,    
                                 embeddings_metadata=None, 
                                 **kwargs)
```

execute this command to launch your tensorboard
```tensorboard --logdir=path_to_your_logs```

![alt text](https://static.javatpoint.com/tutorial/tensorflow/images/tensorflow-tensorboard.png)
<p align="center">TensorBoard</p>

### 4. LearningRateScheduler
This callback is useful for changing the learning rate as the training goes on. For example, you may want to lower the learning rate after some epochs. The LearningRateScheduler can help you with that.

``` 
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
```

``schedule``: this is a function that takes the epoch index and returns a new learning rate.
``verbose``: whether or not to print additional logs.

### 5. CSVLogger
This callback records the training history in a CSV file for each epoch. It tracks the ``epoch``, ``accuracy``, ``loss``, ``val_accuracy``, and ``val_loss`` values. You have to include accuracy in the metrics when you compile the model, or else you will see an execution error.

```
tf.keras.callbacks.CSVLogger(filename, 
                             separator=',', 
                             append=False)
```
The logger takes the ```filename```, ```separator```, and ```append``` as arguments. 'append' decides if the logs are added to an old file or a new file. The CSVLogger callback runs when the on_epoch_end event happens during training. So the logs are saved to a file after each epoch.

### 6. LambdaCallback
```
tf.keras.callbacks.LambdaCallback(on_epoch_begin=None, 
                                  on_epoch_end=None, 
                                  on_batch_begin=None, 
                                  on_batch_end=None,    
                                  on_train_begin=None, 
                                  on_train_end=None, 
                                  **kwargs)
```
All the parameters of this callback expect a function which takes the arguments specified here:
``on_epoch_begin`` and ``on_epoch_end``: epoch, logs
``on_batch_begin`` and ``on_batch_end``: batch, logs
``on_train_begin`` and ``on_train_end``: logs

### 7. RemoteMonitor
This callback is useful when you want to post the logs to an API. This callback can also be mimicked using ``LambdaCallback``.
```
tf.keras.callbacks.RemoteMonitor(root='http://localhost:8000',                
                                   path='/publish/epoch/end/', 
                                   field='data',
                                   headers=None, 
                                   send_as_json=False)
```

``root``: this is the URL.
``path``: this is the endpoint name/path.
``field``: this is the name of the key which will have all the logs.
``header``: the header which needs to be sent.
``send_as_json``: if True, the data will be sent in JSON format.

### 8. BaseLogger & History

These two callbacks are built-in for every Keras model. The ``history`` object is given by ``model.fit``, and has a dictionary with the mean accuracy and loss for each epoch. The ``parameters`` property has the dictionary with the settings for training (``epochs``, ``steps``, ``verbose``). If you use a callback to adjust the learning rate, it will be in the history object too.

### 9. TerminateOnNaN
This callback terminates the training if the loss becomes NaN.

`` tf.keras.callbacks.TerminateOnNaN() ``
