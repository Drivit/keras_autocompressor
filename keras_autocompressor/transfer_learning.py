from pathlib import Path

import numpy as np
import tensorflow as tf


def copy_pretrained_translator(
        base_model: tf.keras.Model,
        connection: tuple,
        architecture_type: str,
    ):
    # Get the template file for the current architecture
    if 'mobilenetv1' in architecture_type:
        translator_file = 'mobilenetv1_skip_from_{}_to_{}_5ep_lr0.001.npy'
    elif 'mobilenetv2' in architecture_type:
        translator_file = 'mobilenetv2_skip_from_{}_to_{}_5ep_lr0.001.npy'
    elif 'efficientnetb5' in architecture_type:
        translator_file = 'efficientnetb5_skip_from_{}_to_{}_5ep_lr0.001.npy'

    # Load translator weights
    module_path = Path(__file__).parent
    translators_path = module_path/'translators'
    translator_checkpoint = translators_path/translator_file.format(*connection)
    translator_weights = np.load(str(translator_checkpoint),allow_pickle=True)

    # Copy the pre-trained weights to the base model
    for idx, layer in enumerate(base_model.layers):
        if 'extended_dim' in layer.name:
            base_model.layers[idx].set_weights(translator_weights.tolist())
            break

    return base_model