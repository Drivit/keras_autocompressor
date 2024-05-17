FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

# Install the keras tuner module
RUN pip3 install keras-tuner==1.1.0

# Install tensorflow-datasets module
RUN pip3 install tensorflow-datasets
