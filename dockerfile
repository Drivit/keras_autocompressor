FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

# Install the keras tuner module
RUN pip3 install keras-tuner

# Install tensorflow-datasets module
RUN pip3 install tensorflow-datasets
