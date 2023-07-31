# Makefile to help automate tasks, such as creating a virtual environment

BASH = bin/bash
TEST_PATH=./tests
VENV_NAME = my_venv
VENV_ACTIVATE = . $(VENV_NAME)/bin/activate

all: create_venv install_dependancies 

create_venv:
	python3 -m pip install --user -U virtualenv
	python3 -m virtualenv $(VENV_NAME)
	
install_dependancies:
	$(VENV_ACTIVATE) && python3 -m pip install -r requirements.txt

jupyter:
	$(VENV_ACTIVATE) && python3 -m ipykernel install --user --name=python3-kernel

download_data:
	@echo "Downloading the dataset"
	$(VENV_ACTIVATE) && python3 src/data_processing/download_dataset.py
	rm data/raw/housing.tgz

prep_csvs:
	$(VENV_ACTIVATE) && python3 src/data_processing/create_csv.py && python3 src/data_processing/create_test_train_csvs.py

prep_data: download_data prep_csvs

run_app: 
	$(VENV_ACTIVATE) && uvicorn app.model_app.server:app
	@echo "The app is running on the localhost, send your pred requests in JSON format to http://127.0.0.1:8000/predict/"

test:
	$(VENV_ACTIVATE) && pytest --verbose --color=yes $(TEST_PATH)

clean_build: 
	rm -rf build dist *.pyc *.pyo *~ __pycache__

build_run_container:
	docker build -t cal_housing_server .
	docker run -p 8000:8000 cal_housing_server
	@echo "The app is running on the container, send your pred requests in JSON format to http://127.0.0.1:8000/predict/"

check_safety_dep:
	safety check -r requirements_prod.txt

help:
	@echo "  all                    : Create a virtual environment, install dependencies"
	@echo "  prep                   : Download the dataset, create CSVs for training & testing"
	@echo "  jupyter                : Install jupyter kernel"
	@echo "  test                   : Run unit tests"
	@echo "  clean_build            : Clean build artifacts"
	@echo "  run_app                : Run the application on the localhost"
	@echo "  build_run_container    : Build the production container and run the inference server on it"
	@echo "  check_safety_dep       : Check for security vulnariebilities in production libraries"