import numpy as np
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

class OccasionalModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        self.model_name_ = kwargs.pop('name', 'unknown')
        self.save_every_k_epochs_ = kwargs.pop('save_every_k_epochs', 100)
        ModelCheckpoint.__init__(self, *args, **kwargs)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.save_every_k_epochs_ == 0 and epoch > 0: 
            print('\n' + '=' * 120 + '\n{}:'.format(self.model_name_))
            super(OccasionalModelCheckpoint, self).on_epoch_end(epoch, logs=logs)


class OccasionalModelTesting(Callback):
    def __init__(self, test_cb=lambda: None, every_k_epochs=100,
                 save_best_only=False, monitor='val_loss'):
        if not hasattr(test_cb, '__call__'): 
            raise RuntimeError('You need to provide callback for testing')
        self.test_cb = test_cb
        self.every_k_epochs = every_k_epochs
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.monitor_op = np.less
        self.best = np.Inf
        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.every_k_epochs == 0: 
            current = logs.get(self.monitor)

            if self.monitor_op(current, self.best) or \
               not self.save_best_only:
                self.best = current

                print('\nTesting model with latest model epoch:{}'.format(epoch))
                self.test_cb(self.model, epoch)
            else:
                print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))


