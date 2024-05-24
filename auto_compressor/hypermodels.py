import tensorflow as tf
import keras_tuner as kt

from .pruning import create_submodel
from .metrics import AccuracyCompression
from .transfer_learning import copy_pretrained_translator

class HyperCompressor(kt.HyperModel):

    def __init__(
            self, 
            max_parameters: int, 
            num_classes:int, 
            tau:float, 
            name:'str|None'=None, 
            tunable:bool=True
        ):
        super().__init__(name, tunable)

        # Save the max_parameters as a class attribute
        self._max_parameters = max_parameters
        
        # Save tau as a class attribute
        self._tau = tau

        # Set the model architecture type
        self._architecture_type = None
        self._backbone_function = None
        self._translators = None
        self._model_input_shape = None

        # Save the number of classes for this classification problem
        self._num_classes = num_classes

    def create_backbone(self, hp: kt.HyperParameters) -> tf.keras.Model:
        '''
        This method creates a backbone for specific supported architectures.
        Basically, the method receives the hyperparameters from a tuner and 
        selects a translator block to generate the pretrained backbone. 
        '''
        # Select the back-bone for the model
        translator = hp.Choice('translator', self._translators)
        translator = tuple(
            int(elem) for elem in translator.split(',')
        )

        # Create the base architecture
        backbone = create_submodel(
            base_model = self._backbone_function(
                input_shape=self._model_input_shape, 
                include_top=False), 
            connection = translator, 
            architecture_type = self._architecture_type
        )
        
        # Copy the pre-trained weights to the base model
        backbone = copy_pretrained_translator(
            base_model = backbone,
            connection = translator,
            architecture_type = self._architecture_type,
        )

        return backbone


    def build(self, hp: kt.HyperParameters) -> tf.keras.Model:
        '''
        This method will be called for each hyperparameters search trail. 
        Building a new model with different hyperparameters according to the
        search space.
        '''
        # Get the backbone
        backbone = self.create_backbone(hp)

        # Freeze the base model
        backbone.trainable = False

        # Create the new top for the network
        x = backbone.output
        x = tf.keras.layers.GlobalAveragePooling2D(
            name='top_gap')(x)
        x = tf.keras.layers.Dense(
            hp.Int('top_fc1_units', 4, 32, step=8, default=4), name='top_fc1')(x)
        x = tf.keras.layers.Dense(
            self._num_classes, name='classifier', activation='softmax')(x)

        # Create the new model
        model = tf.keras.Model(
            inputs=backbone.inputs, 
            outputs=x, 
            name='autosearch'
        )

        # Calculate the compression rate for the proposed metric
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
            metrics=['accuracy', accuracy_compression_metric]
        )

        return model


    

class HyperCompressedMobileNet(HyperCompressor):
    '''
    HyperMobilenet implementation.
    '''

    def __init__(
            self, 
            max_parameters: int, 
            num_classes:int, 
            tau:float=0.8, 
            name=None, 
            tunable=True,
        ):
        # Initialize the base compressor 
        super().__init__(max_parameters, num_classes, tau, name, tunable)

        # Set the model architecture type
        self._architecture_type = 'mobilenetv1'
        self._backbone_function = tf.keras.applications.MobileNet
        self._model_input_shape = (224,224,3)

        # Set the the pre-trained translator connections, in format 'A,B'
        self._translators = (
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
        )

    

class HyperCompressedMobileNetV2(HyperCompressor):
    '''
    HyperMobilenetV2 implementation.
    '''

    def __init__(
            self,
            max_parameters: int, 
            num_classes:int, 
            tau:float=0.8, 
            name=None, 
            tunable=True,
        ):
        # Initialize the base compressor 
        super().__init__(max_parameters, num_classes, tau, name, tunable)

        # Set the model architecture type
        self._architecture_type = 'mobilenetv2'
        self._backbone_function = tf.keras.applications.MobileNetV2
        self._model_input_shape = (224,224,3)

        # Set the the pre-trained submodels connections, in format 'A,B'
        self._translators = (
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
        )


class HyperCompressedEfficientNetB5(HyperCompressor):
    '''
    HyperEfficientNetB5 implementation.
    '''

    def __init__(
            self,
            max_parmeters: int, 
            num_classes:int, 
            tau:float=0.8, 
            name=None, 
            tunable=True,
        ):
        # Initialize the base compressor 
        super().__init__(max_parmeters, num_classes, tau, name, tunable)

        # Set the model architecture type
        self._architecture_type = 'efficientnetb5'
        self._backbone_function = tf.keras.applications.EfficientNetB5
        self._model_input_shape = (456,456,3)

        # Set the the pre-trained submodels connections, in format 'A,B'
        self._translators = (
            '40,396',
            '40,530',
            '114,396',
            '114,530',
            '188,396',
            '188,530',
            '292,396',
            '292,530',
            '395,530',
        )