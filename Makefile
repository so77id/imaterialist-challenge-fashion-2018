# Kaggle imaterialist challenge fashion 2018 Makefile
PROGRAM="Kaggle-imaterialist-fashion-2018"

CPU_REGISTRY_URL=so77id
GPU_REGISTRY_URL=so77id
CPU_VERSION=latest-cpu
GPU_VERSION=latest-gpu
CPU_DOCKER_IMAGE=tensorflow-opencv-py3
GPU_DOCKER_IMAGE=tensorflow-opencv-py3

##############################################################################
############################# Exposed vars ####################################
##############################################################################
# enable/disable GPU usage
GPU=false
# Config file used to experiment
CONFIG_FILE=""
# List of cuda devises
CUDA_VISIBLE_DEVICES=0

#Path to src folder
HOST_CPU_SOURCE_PATH = ""
HOST_GPU_SOURCE_PATH = ""
# Path to dataset
HOST_CPU_DATASETS_PATH = ""
HOST_GPU_DATASETS_PATH = ""
# Path to metada
HOST_CPU_METADATA_PATH = ""
HOST_GPU_METADATA_PATH = ""

##############################################################################
############################# DOCKER VARS ####################################
##############################################################################
# COMMANDS
DOCKER_COMMAND=docker
NVIDIA_DOCKER_COMMAND=nvidia-docker


#HOST VARS
LOCALHOST_IP=127.0.0.1
HOST_TENSORBOARD_PORT=26006
HOST_NOTEBOOK_PORT=28888

#HOST CPU VARS
HOST_CPU_SOURCE_PATH=$(shell pwd)
# TODO make this working for all researcher in this project
HOST_CPU_DATASETS_PATH=/Users/so77id/Desktop/workspace/kaggle/datasets
HOST_CPU_METADATA_PATH=/Users/so77id/Desktop/workspace/kaggle/metadata

#HOST GPU PATHS
HOST_GPU_SOURCE_PATH=$(shell pwd)
HOST_GPU_DATASETS_PATH=/datasets/$(USER)
HOST_GPU_METADATA_PATH=/home/$(USER)/kaggle/imaterialist-challenge-fashion-2018/metadata

#IMAGE VARS
IMAGE_TENSORBOARD_PORT=6006
IMAGE_NOTEBOOK_PORT=8888
IMAGE_SOURCE_PATH=/home/src
IMAGE_DATASETS_PATH=/home/datasets
IMAGE_METADATA_PATH=/home/metadata


# VOLUMES

CPU_DOCKER_VOLUMES = --volume=$(HOST_CPU_SOURCE_PATH):$(IMAGE_SOURCE_PATH) \
				     --volume=$(HOST_CPU_DATASETS_PATH):$(IMAGE_DATASETS_PATH) \
				     --volume=$(HOST_CPU_METADATA_PATH):$(IMAGE_METADATA_PATH) \
				     --workdir=$(IMAGE_SOURCE_PATH)

GPU_DOCKER_VOLUMES = --volume=$(HOST_GPU_SOURCE_PATH):$(IMAGE_SOURCE_PATH) \
				     --volume=$(HOST_GPU_DATASETS_PATH):$(IMAGE_DATASETS_PATH) \
				     --volume=$(HOST_GPU_METADATA_PATH):$(IMAGE_METADATA_PATH) \
				     --workdir=$(IMAGE_SOURCE_PATH)


DOCKER_TENSORBOARD_PORTS = -p $(LOCALHOST_IP):$(HOST_TENSORBOARD_PORT):$(IMAGE_TENSORBOARD_PORT)
DOCKER_JUPYTER_PORTS = -p $(LOCALHOST_IP):$(HOST_NOTEBOOK_PORT):$(IMAGE_NOTEBOOK_PORT)

# IF GPU == false --> GPU is disabled
# IF GPU == true --> GPU is enabled
ifeq ($(GPU), true)
	DOCKER_RUN_COMMAND=$(NVIDIA_DOCKER_COMMAND) run -it --rm --userns=host $(GPU_DOCKER_VOLUMES) $(GPU_REGISTRY_URL)/$(GPU_DOCKER_IMAGE):$(GPU_VERSION)
	DOCKER_RUN_TENSORBOARD_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host $(DOCKER_TENSORBOARD_PORTS) $(GPU_DOCKER_VOLUMES) $(CPU_REGISTRY_URL)/$(CPU_DOCKER_IMAGE):$(CPU_VERSION)
	DOCKER_RUN_JUPYTER_COMMAND=$(NVIDIA_DOCKER_COMMAND) run -it --rm --userns=host $(DOCKER_JUPYTER_PORTS) $(GPU_DOCKER_VOLUMES) $(GPU_REGISTRY_URL)/$(GPU_DOCKER_IMAGE):$(GPU_VERSION)
else
	DOCKER_RUN_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host $(CPU_DOCKER_VOLUMES) $(CPU_REGISTRY_URL)/$(CPU_DOCKER_IMAGE):$(CPU_VERSION)
	DOCKER_RUN_TENSORBOARD_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host $(DOCKER_TENSORBOARD_PORTS) $(CPU_DOCKER_VOLUMES) $(CPU_REGISTRY_URL)/$(CPU_DOCKER_IMAGE):$(CPU_VERSION)
	DOCKER_RUN_JUPYTER_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host $(DOCKER_JUPYTER_PORTS) $(CPU_DOCKER_VOLUMES) $(CPU_REGISTRY_URL)/$(CPU_DOCKER_IMAGE):$(CPU_VERSION)
endif


##############################################################################
############################## CODE VARS #####################################
##############################################################################
#COMMANDS
PYTHON_COMMAND=python3 -m
EXPORT_COMMAND=export
MKDIR_COMMAND=mkdir
BASH_COMMAND=bash
TENSORBOARD_COMMAND=tensorboard
JUPYTER_COMMAND=jupyter
WGET_COMMAND=wget
UNZIP_COMMAND=unzip
RM_COMMAND=rm
MV_COMMAND=mv

# Dataset VARS
DATASET_FOLDER=$(IMAGE_DATASETS_PATH)/kaggle/imaterialist-challenge-fashion-2018
TRAIN_FOLDER=$(DATASET_FOLDER)/train
VALIDATION_FOLDER=$(DATASET_FOLDER)/validation
TEST_FOLDER=$(DATASET_FOLDER)/test
DATA_JSON_URL=http://www.recod.ic.unicamp.br/~mrodriguez/kaggle/imaterialist-challenge-fashion-2018/data.zip

# Metadata VARS
METADATA_FOLDER=$(IMAGE_METADATA_PATH)

#FILES
PROCESS_DATABASE_FILE=utils/dataset_utils/download.sh

CONFIG_FILE=./configs/config.json

CREATE_LIST_FILE=utils.dataset_utils.create_list
MAIN_DATASET_FILE=mains.dataset_main
KERAS_MAIN_FILE=mains.keras_main
KERAS_PREDICT_FILE=mains.keras_predict
KERAS_VOTING_FILE=mains.voting_emsemble_main

# MODEL CHECKPOINTS URLS KERAS
IMAGENET_CHECKPOINTS_FOLDER=./imagenet_checkpoints
INCEPTION_CHECKPOINT_URL=https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_tf_dim_ordering_tf_kernels.h5
INCEPTION_CHECKPOINT_FILENAME=inception-v4_weights_tf_dim_ordering_tf_kernels.h5

RESNET_50_CHECKPOINT_URL=https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5
RESNET_50_CHECKPOINT_FILENAME=resnet50_weights_tf_dim_ordering_tf_kernels.h5

DENSENET_121_CHECKPOINT_URL=http://www.recod.ic.unicamp.br/\~mrodriguez/densenet121_weights_tf.h5
DENSENET_121_CHECKPOINT_FILENAME=densenet121_weights_tf.h5
#############################################################################
############################ CODE COMMANDS ###################################
##############################################################################
all: help

setup s: setup-checkpoints
	@echo "[Setup] Finish.."

setup-folder sf:
	@echo "[Setup] Setup folders.."
	@$(MKDIR_COMMAND) -p $(TRAIN_FOLDER)
	@$(MKDIR_COMMAND) -p $(VALIDATION_FOLDER)
	@$(MKDIR_COMMAND) -p $(TEST_FOLDER)
	@$(MKDIR_COMMAND) -p $(METADATA_FOLDER)

process-dataset pd:
	@echo "[Dataset Processing] Downloading.."
	@$(WGET_COMMAND) $(DATA_JSON_URL) -P $(DATASET_FOLDER)
	@$(UNZIP_COMMAND) $(DATASET_FOLDER)/data.zip -d $(DATASET_FOLDER)

	@$(BASH_COMMAND) $(PROCESS_DATABASE_FILE)
	@$(PYTHON_COMMAND) $(CREATE_LIST_FILE) -c $(CONFIG_FILE)

	@echo "[Dataset Processing] Deleting files.."
	@$(RM_COMMAND) $(DATASET_FOLDER)/data.zip
	@$(RM_COMMAND) $(DATASET_FOLDER)/train.json
	@$(RM_COMMAND) $(DATASET_FOLDER)/validation.json
	@$(RM_COMMAND) $(DATASET_FOLDER)/test.json


setup-checkpoints sch:
	@echo "[Setup Checkpoints] Downloading.."
	@$(MKDIR_COMMAND) -p $(IMAGENET_CHECKPOINTS_FOLDER)
	@$(WGET_COMMAND) $(INCEPTION_CHECKPOINT_URL)
	@$(MV_COMMAND) $(INCEPTION_CHECKPOINT_FILENAME) $(IMAGENET_CHECKPOINTS_FOLDER)

	@$(WGET_COMMAND) $(RESNET_50_CHECKPOINT_URL)
	@$(MV_COMMAND) $(RESNET_50_CHECKPOINT_FILENAME) $(IMAGENET_CHECKPOINTS_FOLDER)

	@$(WGET_COMMAND) $(DENSENET_121_CHECKPOINT_URL)
	@$(MV_COMMAND) $(DENSENET_121_CHECKPOINT_FILENAME) $(IMAGENET_CHECKPOINTS_FOLDER)

test-dataset-loader tdl:
	@echo "[Test dataset loader] Testing.."
	@$(PYTHON_COMMAND) $(MAIN_DATASET_FILE) -c $(CONFIG_FILE)

train-keras tk:
	@echo "[Train Keras] Trainning.."
	@$(EXPORT_COMMAND) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)
	@$(PYTHON_COMMAND) $(KERAS_MAIN_FILE) -c $(CONFIG_FILE)

predict-keras pk:
	@echo "[Predict Keras] Predicting.."
	@$(EXPORT_COMMAND) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)
	@$(PYTHON_COMMAND) $(KERAS_PREDICT_FILE) -c $(CONFIG_FILE)


voting-emsemble ve:
	@echo "[Predict Keras with voting emsemble] Predicting.."
	@$(EXPORT_COMMAND) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)
	@$(PYTHON_COMMAND) $(KERAS_VOTING_FILE) -c $(CONFIG_FILE)

tensorboard tb:
	@echo "[Tensorboard] Running Tensorboard"
	@$(TENSORBOARD_COMMAND) --logdir=$(IMAGE_METADATA_PATH) --host 0.0.0.0

jupyter jp:
	@echo "[Jupyter] Running Jupyter lab"
	@$(JUPYTER_COMMAND) lab --allow-root

#############################################################################
########################### DOCKER COMMANDS ##################################
##############################################################################

run-test rtm: docker-print
	@$(DOCKER_RUN_COMMAND)

run-tensorboard rt: docker-print
	@$(DOCKER_RUN_TENSORBOARD_COMMAND)  bash -c "make tensorboard IMAGE_METADATA_PATH=$(IMAGE_METADATA_PATH)"; \
	status=$$?

run-jupyter rj: docker-print
	@$(DOCKER_RUN_JUPYTER_COMMAND)  bash -c "make jupyter"; \
	status=$$?

run-setup rpd: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make setup"; \
	status=$$?

run-dataset-loader rdl: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make test-dataset-loader CONFIG_FILE=$(CONFIG_FILE)"; \
	status=$$?

run-train-keras rtk: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make train-keras CONFIG_FILE=$(CONFIG_FILE) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"; \
	status=$$?

run-predict-keras rpk: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make predict-keras CONFIG_FILE=$(CONFIG_FILE) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"; \
	status=$$?

run-voting_emsemble rvek: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make voting-emsemble CONFIG_FILE=$(CONFIG_FILE) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"; \
	status=$$?

#PRIVATE
docker-print psd:
ifeq ($(GPU), true)
	@echo "[GPU Docker] Running gpu docker image..."
else
	@echo "[CPU Docker] Running cpu docker image..."
endif


help:
	@echo ""
	@echo "Makefile for $(PROGRAM)"
	@echo ""
	@echo "DOCKER COMMANDS"
	@echo "make [run-tensorboard | run-jupyter | run-setup | run-test]"
	@echo "-----------------------------------------------------------------------------------"
	@echo "   - run-tensorboard : Run docker image with tensorboard command"
	@echo "   - run-jupyter     : Run docker image with jupyter command"
	@echo "   - run-setup       : Run docker image with setup command"
	@echo "   - run-test        : Run docker image in test mode"
	@echo ""
	@echo "EXPERIMENT COMMANDS"
	@echo "make [setup | tensorboard | jupyter]"
	@echo "-----------------------------------------------------"
	@echo "   - setup           : Setup folder and download dataset"
	@echo "   - tensorboard     : Run tensorboard"
	@echo "   - jupyter         : Run jupyter lab"

.PHONY: run-tensorboard run-jupyter run-setup run-test setup tensorboard jupyter
