### Detecting abnormalities in X-Ray images using Neural Networks

------------

This is a bachelor work by Uladzislau Yorsh. On the example of participating in RA2 Challenge the joint damage assessment model was created.

This repository includes:
- Colab notebooks used to train the detection models and a one to train the assessment models (for testing)
- The bachelor work LaTeX sources with images
- Docker container files
- Training logs and several example images
- Manual annotations used for joint detectors training

What this repository does not include:
- The data. They have to be downloaded from the RA2 Challange page by a registered user
- The pre-trained detection models for the docker container. They can be downloaded from the CVUT FIT faculty Google Drive: [Link](https://drive.google.com/open?id=1JIwfKTvv-ftfmyQbhgtUjo-wNpK-LfiR)

How to get the data:
- First of all, you have to be registered for the RA2 Challenge: [Link](https://www.synapse.org/#!Synapse:syn20545111/wiki/594083 "Link")
- At the challenge wiki pages, go to the "Data" page and use the wget command to download the public set

How to build the docker container:
- Download the four pre-trained detectors from the link above
- Go to the docker context folder---it has to contain the Dockerfile, the script.py file and the "weights" directory with a weights.h5 file
- Create a trained_models directory at the docker context folder and place the four downloaded weight sets inside
- Follow the instructions at the RA2 Challenge wiki: [Link](https://www.synapse.org/#!Synapse:syn20545111/wiki/597249 "Link"). Proceed from point 4.

According to the wiki pages, the scoring container will be publically available after the challenge end.

Notes about notebooks:
- Public data sets were changed multiple times during the challenge. Notebooks dedicated to detection models refer to some distinct files through .json label files and 'force_train' lists, and certain files may be absent in new publicly available sets.
- Detection models notebooks refer to the aligned data sets (with right images flipped), not to the original ones.
