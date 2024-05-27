import numpy as np
import tensorflow as tf

def create_submodel(
    base_model: tf.keras.Model, 
    connection: tuple, 
    architecture_type: str,
):
    """This function creates a sub-model from a given base model. The created 
    submodel includes a translator block [[1](https://www.mdpi.com/2076-3417/12/10/4945)],
    creating a connection between layers A and B.

    Args

    base_model: `tf.keras.Model`
       base model loaded in memory.

    connection : `tuple`
        the indexes for the translator connection.

    architecture_type : `str`
        base architecture type, either 'mobilenet', 'mobilenetv2' or 
        'efficientnetb5'.


    Returns

        A `tf.keras.Model`


    References
    [1] [Neuroplasticity-Based Pruning Method for Deep Convolutional Neural Networks](https://www.mdpi.com/2076-3417/12/10/4945)
    """    
    # Get the spatial dimensions for the connection
    target_output_shape = base_model.layers[connection[1]].input_shape
    actual_output_shape = base_model.layers[connection[0]].output_shape

    # Get the A's output tensor from the base model
    x = base_model.layers[connection[0]].output

    # Check if A's output needs a spatial transformation, else the pooling 
    # layer is not included in the model 
    if target_output_shape[1:3] != actual_output_shape[1:3]:
        if 'efficientnetb5' in architecture_type:
            # For EfficientNets
            s_size = np.ceil(actual_output_shape[1] / target_output_shape[1])
            k_size = actual_output_shape[1] - s_size*(target_output_shape[1] - 1)
            s_size = (s_size, s_size)
            k_size = (k_size, k_size)
        else:
            # For MobileNets and VGGNets
            size = actual_output_shape[1] / target_output_shape[1]
            s_size = (np.floor(size), np.floor(size))
            k_size = s_size


        # Transform the output of the layer #DEBUG
        # print ('Target: ', target_output_shape)
        # print ('Actual output: ', actual_output_shape)
        # print ('Calculated window and stride: ', size)

        # Reduce spatial dimension
        x = tf.keras.layers.MaxPool2D(
            pool_size=k_size, 
            strides=s_size, 
            name='transform_pool',
        )(x)

    # Add an extra layer to augment the volume depth; this layer will be the
    # only one unfreeze during the training
    x = tf.keras.layers.Conv2D(
        target_output_shape[-1], # n_kernels
        kernel_size=(1,1),
        name='extended_dim',
        activation='relu',
    )(x)

    # ---- Reconnection phase ----
    # Extract the input tensors for all the layers in the model
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
    for layer in base_model.layers:
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in network_dict['input_layers_of']:
                    network_dict['input_layers_of'].update(
                            {layer_name: [layer.name]})
                else:
                    network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set new B's input tensor, i.e. the translator block
    network_dict['new_output_tensor_of'].update(
            {base_model.layers[connection[1]-1].name: x})

    # Reconect the remaning layers
    model_outputs = []
    for layer in base_model.layers[connection[1]:]:
            # Determine input tensors
            layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                    for layer_aux in network_dict['input_layers_of'][layer.name]]
            
            if len(layer_input) == 1:
                layer_input = layer_input[0]

            x = layer(layer_input)

            # Set new output tensor (the original one, or the one of the inserted
            # layer)
            network_dict['new_output_tensor_of'].update({layer.name: x})

            # Save tensor in output list if it is output in initial model
            if layer.name in base_model.output_names:
                model_outputs.append(x)

    # ---- Generate the submodel ----
    model = tf.keras.Model(
        inputs=base_model.inputs, 
        outputs=model_outputs
    )
    
    return model