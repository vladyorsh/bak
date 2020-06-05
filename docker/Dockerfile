## Start from this Docker image
FROM ubuntu

# Install python3 and regular dependencies
RUN apt-get update -y && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install python3 -y && \
    apt-get install -y git wget && \
    apt-get install -y libpython3-dev && \
    apt-get install -y python-pip libjpeg-dev && \
    apt-get install -y python3-pip && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    pip3 install --upgrade pip && \
    pip3 install opencv-python pandas numpy && \
    pip3 install scikit-learn

#For Matterport MRCNN support
RUN pip3 install scikit-image && pip3 install ipython

# Install tensorflow and keras to train and run models
RUN pip3 install tensorflow-gpu==1.15 keras==2.2.5

#Instal Matterport MRCNN
RUN git clone https://github.com/matterport/Mask_RCNN.git && cd Mask_RCNN && python3 setup.py install && pip show mask-rcnn

# Required: Create /train /test and /output directories
RUN mkdir /train && mkdir /test && mkdir /output

# Create directories for auxilary files
RUN mkdir /trained_models && mkdir /weights && mkdir /output/logs

# Required for GPU: run.sh defines PATHs to find GPU drivers, see run.sh for specific commands
COPY run.sh /run.sh

COPY trained_models /trained_models
COPY weights /weights

COPY test.py /usr/local/bin/test.py
COPY script.py /usr/local/bin/ra2.py

# Make model and runfiles executable
#RUN chmod 775 /usr/local/bin/model.R
RUN chmod 775 /run.sh && chmod 775 /usr/local/bin/test.py && chmod 775 /usr/local/bin/ra2.py

RUN chmod 775 /weights && chmod 775 /output && chmod 775 /trained_models

# This is for the virtualenv defined above, if not using a virtualenv, this is not necessary
#RUN chmod 755 /root #to make virtualenv accessible to singularity user

# Required: define an entrypoint. run.sh will run the model for us, but in a different configuration
# you could simply call the model file directly as an entrypoint
ENTRYPOINT ["/bin/bash", "/run.sh"]
