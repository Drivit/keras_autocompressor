# Keras auto-compressor
This is the official repository for the model auto-compression methodology presented in the paper "Auto-compression Transfer Learning Methodology for Deep Convolutional Neural Networks".

We used a [bioinspired pruning strategy](https://www.mdpi.com/2076-3417/12/10/4945) that can delete entire layers or blocks from the network architecture by using pretrained translator layers. This virtually offers a group of sub-architectures with various compression rates, leaving the pruning step as another hyperparameter for tuning.

For further technical details, please refer to the paper.

<!-- ## Installation
Our auto-compressor is available on the Python Package Index ([PyPI](https://pypi.org/)) as `keras-autocompressor`, install this module with the following command.
``` shell
pip install keras-autocompressor
```
To use the `keras-autocompressor` module, the [tensorflow](https://pypi.org/project/tensorflow/) and [keras-tuner](https://pypi.org/project/keras-tuner/) packages are included as dependencies. -->

## Dependencies
This module was designed for [Tensorflow](https://www.tensorflow.org/) and the [keras-tuner](https://keras.io/keras_tuner/) toolkit. We strongly recommend using the following package versions (used in the development phase) to test this module and avoid compatibility issues.

- tensorflow >= 2.4, < 2.16 
- keras-tuner >= 1.1.0 

We'll be including further notes about the compatibility with newer package versions because Tensorflow 2.16.x (due to Keras 3.x) is not currently supported.

### Docker
Here we included a [dockerfile](./dockerfile) to replicate the image that we used for this implementation.

> NOTE: this docker image includes the [tensorflow-datasets](https://www.tensorflow.org/datasets/overview) module. However it is not required by our module to operate properly, it is only included for the notebook [examples](./test/).


## Usage
To execute the auto-compression task you can use any hyperparameter seach algorithm ([tuner](https://keras.io/api/keras_tuner/tuners/)) included in the keras-tuner module.  For our experiments and examples we used the [hyperband tuner](https://keras.io/api/keras_tuner/tuners/hyperband/).


### Hypermodels
In the pruning-search process you can select the [MobileNet](https://arxiv.org/abs/1704.04861), [MobileNetV2](https://arxiv.org/abs/1801.04381) and, [EfficientNetB5](https://arxiv.org/abs/1905.11946); as the backbone architectures. We provided our custom [HyperModels](./src/keras_autocompressor/hypermodels.py), wrapping these architectures with new top structures created according to a given classification task.


This is an example of using a MobileNetV2 backbone for the pruning-search process with a dataset of 5 classes.

``` python
>>> from keras_autocompressor.hypermodels import HyperCompressedMobileNetV2
>>>
>>> hyper_model = HyperCompressedMobileNetV2(
>>>    max_parameters=2.3E6,
>>>    num_classes=5,
>>>    tau=0.4)

```

Here the `max_parameter` referes to the total number of parameters in a model created with the classical transfer learning approach. We suggest to create the base model once  to get this number, however, you can use and approximation as in this example.

### Tuner search
Once the Hypermodel is created, the pruning-search process should be executed. Here is an example for our previously defined hypermodel with the Hyperband tuner.

``` python
>>> import keras-tuner as kt
>>>
>>> # Create the tuner object for our search
>>> mobilenetv2_compressor = kt.Hyperband(
>>>     hyper_model,
>>>     max_epochs=50,
>>>     objective=kt.Objective("val_acc_comp", direction="max"),
>>>     directory='./tuner_logs/mobilenetv2/',
>>>     project_name='sample_dataset')
>>>
>>> # Run the hyperparameters search + auto compression
>>> mobilenetv2_compressor.search(ds_train, validation_data=ds_val)
```

### Custom model definition
Even when our hypermodels have their own definitions, you can modify the structure and keep using the pruning-search process. Assuming that you want to change the hypermodels' top, we encorage you to use our [hypermodels](./src/keras_autocompressor/hypermodels.py) as base classes and override the `build()` method. 

For example, consider chaging the model optimizer as the Stochastic Gradient Descent (SGD) instead of using [Adam](https://arxiv.org/abs/1412.6980),  then your custom hypermodel should be defined as in [this example](./test/Custom_MobileNetV2_Compressor.ipynb).

## Citation
Please include the following citation if you are using our methodology for your works or experimentation.
``` bibtex
@article{CAMACHO2025129661,
    title = {Auto-compression transfer learning methodology for deep convolutional neural networks},
    journal = {Neurocomputing},
    volume = {630},
    pages = {129661},
    year = {2025},
    issn = {0925-2312},
    doi = {https://doi.org/10.1016/j.neucom.2025.129661},
    url = {https://www.sciencedirect.com/science/article/pii/S0925231225003339},
    author = {J.D. Camacho and Carlos Villaseñor and Javier Gomez-Avila and Carlos Lopez-Franco and Nancy Arana-Daniel},
}
```