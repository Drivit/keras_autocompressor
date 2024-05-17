import tensorflow as tf

class AccuracyCompression(tf.keras.metrics.Metric):
    '''
    '''
    def __init__(
        self, 
        compression_rate:float, 
        tau=0.8, 
        name='accuracy_compression',
        **kwargs
      ):
        # Call the base class init
        super().__init__(name=name, **kwargs)

        # Create the placeholders for the metric components
        self.accuracy_compression = self.add_weight(
            name='acc_comp', 
            initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

        self.tau = tf.cast(tau, self.dtype)
        self.compression_rate = tf.cast(compression_rate, self.dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        matches = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
        self.count.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(matches), dtype=self.dtype))

    def result(self):   
        # Compute the accuracy-compression metric at the end of the batch 
        acc_comp = tf.add(
            # accuracy component
            tf.multiply(
                self.tau, 
                tf.truediv(self.count, self.total)),  
            # compression component
            tf.multiply(
                tf.subtract(tf.constant(1.0), self.tau), 
                self.compression_rate))
        
        self.accuracy_compression.assign(acc_comp)
        return self.accuracy_compression

    def reset_states(self):
        self.accuracy_compression.assign(0)
        self.total.assign(0)
        self.count.assign(0)