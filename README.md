###Detecting abnormalities in X-Ray images using Neural Networks

------------

This is a bachelor work by Uladzislau Yorsh. On the example of participating in RA2 Challenge the joint damage assessment model was created.

This repository includes:
- Colab notebooks used to train the detection models and a one to train the assessment models (for testing)
- The bachelor work LaTeX sources with images
- Docker container files
- Training logs and several example images

What this repository does not include:
- The data. They have to be downloaded from the RA2 Challange page by a registered user
- The pre-trained detection models for the docker container. They can be downloaded from the faculty Google Drive: [will be available after uploading]

How to get the data:
- First of all, you have to be registered for the RA2 Challenge: [Link](https://www.synapse.org/#!Synapse:syn20545111/wiki/594083 "Link")
- At the challenge wiki pages, go to the "Data" page and use the wget command to download the public set

How to build the docker container:
- Download the four pre-trained detectors from the link above
- Go to the docker context folder---it has to contain the Dockerfile, the script.py file and the "weights" directory with a weights.h5 file
- Create a trained_models directory at the docker context folder and place the four downloaded weight sets inside
- Follow the instructions at the RA2 Challenge wiki: [Link](https://www.synapse.org/#!Synapse:syn20545111/wiki/597249 "Link"). Proceed from point 4.

According to the wiki pages, the scoring container will be publically available after the challenge end.