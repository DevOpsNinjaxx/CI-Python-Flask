[![CircleCI](https://dl.circleci.com/status-badge/img/gh/Fouty7/project-ml-microservice-kubernetes/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/Fouty7/project-ml-microservice-kubernetes/tree/main)

## Project Overview

This project operationalizes a Python flask app—in a provided file, `app.py`—that serves out predictions (inference) about housing prices through API calls. It uses a pre-trained, `sklearn` model that has been trained to predict housing prices in Boston according to several features, such as average rooms in a home and data about highway access, teacher-to-pupil ratios, and so on. You can read more about the data, which was initially taken from Kaggle, on [the data source site](https://www.kaggle.com/c/boston-housing). This project could be extended to any pre-trained machine learning model, such as those for image recognition and data labeling.

### Getting Started / Summary

* Install python 3.7. To get it on Ubuntu you can follow the instructions here: 
  https://www.linuxcapable.com/how-to-install-python-3-7-on-ubuntu-20-04-lts/
* Install Docker 
* Install Kubernetes via minikube
* Build Container using docker
* Configure kubernetes to run locally
* Deploy a container using Kubernetes and make a prediction

---

## Setup the Environment

* Create a virtualenv with Python 3.7 and activate it. Refer to this link for help on specifying the Python version in the virtualenv. 
```bash
python3 -m pip install --user virtualenv
# You should have Python 3.7 available in your host. 
# Check the Python path using `which python3`
# Use a command similar to this one:
python3 -m virtualenv --python=<path-to-Python3.7> .devops
source .devops/bin/activate
```
* Run `make install` to install the necessary dependencies

### Running `app.py`

1. Standalone:  `python app.py`
2. Run in Docker:  `./run_docker.sh`
2. Start minikube:  `minikube start`
3. Run in Kubernetes:  `./run_kubernetes.sh`

### Kubernetes Steps

* Setup and Configure Docker locally
* Setup and Configure Kubernetes locally
* Create Flask app in Container using `./run_docker.sh`
* Start Kubernetes cluster `minikube start`
* Deploy Cointainer in Kubernetes using via kubectl by running `./run_kubernetes.sh`
