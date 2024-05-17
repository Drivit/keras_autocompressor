import tensorflow as tf
import keras_tuner as kt

from .pruning import create_submodel
from .metrics import AccuracyCompression
from .transfer_learning import copy_pretrained_translator

class HyperAutoCompressor(kt.HyperModel):

    def __init__(
            self, 
            max_parmeters: int, 
            num_classes:int, 
            tau=0.8, 
            name=None, 
            tunable=True
        ):
        super().__init__(name, tunable)

        # Save the max_parameters as a class attribute
        self._max_parameters = max_parmeters
        
        # Save tau as a class attribute
        self._tau = tau

        # Set the model architecture type
        self._architecture_type = None
        self._backbone_func = None
        self._translators = None
        self._model_input_shape = None

        # Save the number of classes for this classification problem
        self._num_classes = num_classes


    def build(self, hp: kt.HyperParameters):
        '''
        This method will be called for each hyperparameters search trail. 
        Building a new model with different hyperparameters according to the
        search space.
        '''
        # Select the back-bone for the model
        translator = hp.Choice('translator', self._translators)
        translator = tuple(
            int(elem) for elem in translator.split(',')
        )

        # Create the base architecture
        base_model = create_submodel(
            base_model = self._backbone_func(
                input_shape=self._model_input_shape, 
                include_top=False), 
            connection = translator, 
            architecture_type = self._architecture_type
        )
        
        # Copy the pre-trained weights to the base model
        base_model = copy_pretrained_translator(
            base_model = base_model,
            connection = translator,
            architecture_type = self._architecture_type,
        )

        # Freeze the base model
        base_model.trainable = False

        # Create the new top for the network
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D(
            name='newtop_gap')(x)
        x = tf.keras.layers.Dense(
            hp.Int('top_fc1_units', 4, 32, step=8, default=4), name='top_fc1')(x)
        x = tf.keras.layers.Dense(
            self._num_classes, name='classifier', activation='softmax')(x)

        model = tf.keras.Model(
            inputs=base_model.inputs, 
            outputs=x, 
            name='autosearch'
        )

        # Calculate the rate compression for the proposed metric
        actual_params = model.count_params()
        params_rate = actual_params / self._max_parameters
        compression_rate = 1 - params_rate

        # Create the custom metric for this model
        accuracy_compression_metric = AccuracyCompression(
            name='acc_comp', 
            compression_rate=compression_rate, 
            tau=self._tau
        )
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float('lr', 1E-5, 1E-2, sampling='log')), 
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy', accuracy_compression_metric])

        return model
    

class HyperAutoCompressMobileNet(kt.HyperModel):
    '''
    HyperMobilenet implementation.
    '''

    # Set the the pre-trained translator connections, in format 'A,B'
    translators = [
        '3,73',
        '3,80',
        '9,73',
        '9,80',
        '16,73',
        '16,80',
        '22,73',
        '22,80',
        '29,73',
        '29,80',
        '35,73',
        '35,80',
        '42,73',
        '42,80',
        '48,73',
        '48,80',
        '54,73',
        '54,80',
        '60,73',
        '60,80',
        '66,73',
        '66,80',
        '72,80',
    ]

    def __init__(
            self, 
            max_parmeters: int, 
            num_classes:int, 
            tau=0.8, 
            name=None, 
            tunable=True
        ):
        super().__init__(name, tunable)

        # Save the max_parameters as a class attribute
        self.max_parameters = max_parmeters
        
        # Save tau as a class attribute
        self.__tau = tau

        # Set the model architecture type
        self.__architecture_type = 'mobilenetv1'

        # Save the number of classes for this classification problem
        self.__num_classes = num_classes

    def build(self, hp: kt.HyperParameters):
        '''
        This method will be called for each hyperparameters search trail. 
        Building a new model with different hyperparameters according to the
        search space.
        '''
        # Select the back-bone for the model
        translator = hp.Choice('base_model', self.__class__.translators)
        translator = tuple(
            int(elem) for elem in translator.split(',')
        )

        # Create the base architecture
        base_model = create_submodel(
            base_model = tf.keras.applications.MobileNet(
                input_shape=(224,224,3), 
                include_top=False), 
            connection = translator, 
            architecture_type = self.__architecture_type
        )
        
        # Copy the pre-trained weights to the base model
        base_model = copy_pretrained_translator(
            base_model = base_model,
            connection = translator,
            architecture_type = self.__architecture_type,
        )

        # Freeze the base model
        base_model.trainable = False

        # Create the new top for the network
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D(
            name='newtop_gap')(x)
        x = tf.keras.layers.Dense(
            hp.Int('fc_units', 4, 32, step=8, default=4), name='newtop_fc1')(x)
        x = tf.keras.layers.Dense(
            self.__num_classes, name='classifier', activation='softmax')(x)

        model = tf.keras.Model(
            inputs=base_model.inputs, 
            outputs=x, 
            name='autosearch'
        )

        # Calculate the rate compression for the proposed metric
        actual_params = model.count_params()
        params_rate = actual_params / self.max_parameters
        compression_rate = 1 - params_rate

        # Create the custom metric for this model
        accuracy_compression_metric = AccuracyCompression(
            name='acc_comp', 
            compression_rate=compression_rate, 
            tau=self.__tau
        )
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float('lr', 1E-5, 1E-2, sampling='log')), 
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy', accuracy_compression_metric])

        return model


class HyperAutoCompressMobileNetV2(kt.HyperModel):
    '''
    HyperMobilenetV2 implementation.
    '''

    # Set the the pre-trained submodels connections, in format 'A,B'
    translators = [
        '8,143',
        '8,151',
        '17,143',
        '17,151',
        '26,143',
        '26,151',
        '35,143',
        '35,151',
        '44,143',
        '44,151',
        '53,143',
        '53,151',
        '62,143',
        '62,151',
        '71,143',
        '71,151',
        '80,143',
        '80,151',
        '89,143',
        '89,151',
        '97,143',
        '97,151',
        '106,143',
        '106,151',
        '115,143',
        '115,151',
        '124,143',
        '124,151',
        '133,143',
        '133,151',
        '142,151',
    ]

    def __init__(
            self, 
            max_parmeters: int, 
            num_classes:int, 
            tau=0.8, 
            name=None, 
            tunable=True
        ):
        super().__init__(name, tunable)

        # Save the max_parameters as a class attribute
        self.max_parameters = max_parmeters
        
        # Save tau as a class attribute
        self.__tau = tau

        # Set the model architecture type
        self.__architecture_type = 'mobilenetv2'

        # Save the number of classes for this classification problem
        self.__num_classes = num_classes

    def build(self, hp):
        '''
        This method will be called for each hyperparameters search trail. 
        Building a new model with different hyperparameters according to the
        search space.
        '''
        # Select the back-bone for the model
        translator = hp.Choice('base_model', self.__class__.translators)
        translator = tuple(
            int(elem) for elem in translator.split(',')
        )

        # Create the base architecture
        base_model = create_submodel(
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224,224,3), 
                include_top=False), 
            connection = translator, 
            architecture_type = self.__architecture_type
        )
        
        # Copy the pre-trained weights to the base model
        base_model = copy_pretrained_translator(
            base_model = base_model,
            connection = translator,
            architecture_type = self.__architecture_type,
        )

        # Freeze the base model
        base_model.trainable = False

        # Create the new top for the network
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D(
            name='newtop_gap')(x)
        x = tf.keras.layers.Dense(
            hp.Int('fc_units', 4, 32, step=8, default=4), name='newtop_fc1')(x)
        x = tf.keras.layers.Dense(
            self.__num_classes, name='classifier', activation='softmax')(x)

        model = tf.keras.Model(
            inputs=base_model.inputs, 
            outputs=x, 
            name='autosearch'
        )

        # Calculate the rate compression for the proposed metric
        actual_params = model.count_params()
        params_rate = actual_params / self.max_parameters
        compression_rate = 1 - params_rate

        # Create the custom metric for this model
        accuracy_compression_metric = AccuracyCompression(
            name='acc_comp', 
            compression_rate=compression_rate, 
            tau=self.__tau
        )
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float('lr', 1E-5, 1E-2, sampling='log')), 
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy', accuracy_compression_metric])

        return model
    

class HyperAutoCompressEfficientNetB5(kt.HyperModel):
    '''
    HyperEfficientNetB5 implementation.
    '''

    # Set the the pre-trained submodels connections, in format 'A,B'
    submodels = [
        '40,396',
        '40,530',
        '114,396',
        '114,530',
        '188,396',
        '188,530',
        '292,396',
        '292,530',
        '395,530',
    ]

    max_params_road_damage =  28579194 # 3 classes
    max_params_flowers =  28579260 # 5 classes
    max_params_caltech_101 =  28582428 # 101 classes

    def __init__(self, name=None, tunable=True, alpha=0.8):
        super().__init__(name, tunable)
        self.alpha = alpha

    def build(self, hp):
        '''
        '''
        # Select the back-bone for the model
        skip_conn = hp.Choice('base_model', self.__class__.submodels)
        skip_conn = tuple(int(elem) for elem in skip_conn.split(','))

        # Create the base architecture
        base_model = create_submodel(
            tf.keras.applications.EfficientNetB5(input_shape=(456,456,3), include_top=False), 
            skip_conn, 
            'efficientnetb5_skip_from_{}_to_{}')
        
        # Copy the pre-trained weights to the base model
        base_model = copy_pretrained_translator(
            base_model,
            'efficientnetb5',
            skip_conn,
        )

        # Freeze the base model
        base_model.trainable = False

        # Create the new top for the network
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D(name='newtop_gap')(x)
        x = tf.keras.layers.Dense(hp.Int('fc_units', 4, 32, step=8, default=4), name='newtop_fc1')(x)
        x = tf.keras.layers.Dense(hp.get('n_classes'), name='classifier', activation='softmax')(x)

        model = tf.keras.Model(inputs=base_model.inputs, outputs=x, name='autosearch')

        # Calculate the rate compression for the proposed metric
        actual_params = model.count_params()
        if hp.get('n_classes') == 3:
            params_ratio = actual_params / self.__class__.max_params_road_damage
        if hp.get('n_classes') == 5:
            params_ratio = actual_params / self.__class__.max_params_flowers
        if hp.get('n_classes') == 101:
            params_ratio = actual_params / self.__class__.max_params_caltech_101

        compression_ratio = 1 - params_ratio
        acc_compression = AccuracyCompression('acc_comp', compression_ratio, self.alpha)

        model.compile(optimizer=sKAdam(hp.Float('lr', 1E-5, 1E-2, sampling='log')), 
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy', acc_compression])

        return model