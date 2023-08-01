## 1. Requirements
+ python>=3.7.0

## 2. Installation 
The makefile included with this project contains targets that help to automate several tasks.

1. Clone this repository & go to the project directory
```bash
cd take-home-challenge
```

2. Then create a virtual environment, install its dependancies:
```bash
make all
```
```bash
# You can manually activate it with the following line, this is not necessary if you're only running the targers with the "make" command
source my_venv/bin/activate
```
3. Download the California housing dataset and create train and test CSV files
```bash
make prep_data
```

## 3. Running the application
There are two ways to run the application:
1. Running the python inference server script directly
```bash
make run_app
```

2. Running the python inference server in a docker container
```bash
make build_run_container
```

+ The app accepts inference requests in JSON format at "http://127.0.0.1:8000/predict/" 
+ You can test it in multiple ways, such as:
    1. Run a post request with a json file, such as this command:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d @app/model_app/exp.json http://127.0.0.1:8000/predict/
    ```
    2. Run with the sample JSON file located in app/model_app/exp.json
    ```bash
    python app/model_app/client_json.py
    ```
    3. Run with the sample JSON variable
    ```bash
    python app/model_app/client_sample.py
    ```
    + Don't forget to activate the environment manually for steps 2 & 3 :wink: .
## 4. Training the models
+ For training, the generated CSV file "data/csv/CaliforniaHousing/california_housing_train.csv" is used.
+ To train once with a LinearRegression model, you can run this:
    ```bash
    python src/train.py
    ```
+ To train with a WandB sweep to choose the best hyperparameters:
    ```bash
    python src/train.py --sweep
    ```
    The sweep configuration is located in 'config/sweep_config.py', currently it is designed to run with a GradientBoosting model with different hyperparameters in grid fashion.
+ The training expirements, including model & dataset artifcats, are logged on WandB.
## 5. Inference
+ For inference, the generated CSV file "data/csv/CaliforniaHousing/california_housing_test.csv" is used by default.
+ To run the inference script, run the following:
    ```bash
    python src/inference.py
    ```
## 6. Running the unit tests
To run the unit tests run the following command:
```bash
make test
```
## 7. Cleaning the build artifacts
```bash
make clean_build
```
## 8. Check for security vulnariebilities in production libraries
```bash
make check_safety_dep
```

## 9. Process and decisions
+ **Why you chose the particular dataset and model**

    1. The feature and target variables are relatable & understood by most people. The dataset does not contain complex quantities that might be hard to understand.
    2. The dataset is not too big that it requires heavy computations and/ or long training times, which was required.
    3. The dataset requires some preprocessing steps, such 
    handling missing data, feature scaling, feature engineering.

+ **Any challenges you faced and how you overcame them**

    The most challenging part for me was to do this project as well as possible, without over-engineering it. 

+ **How your application can be updated or maintained in the future**

    There are many ways to update and maintain this application in the future, such as the following:

    1. There are many models, hyperparametrs, dataset splits, feature combinations, tests, deployment methods for both the regular app in "app/model_app/server.py" as well as the containerized app to be tested. However, due to the lack of time and scope of this project this was not possible.

    2. Some tools, such as GitHub Actions and bitbcuket pipelines, can be used to automate the CI/CD pipelines, making it faster to build, test, and deploy the code.
    
    3. Also some git pre-commit hook scripts, such as  can be used to detect simple issues on every commit automatically.
    
    4. Other models can be trained and better performance might be achievable. The containerized app can be deployed for example with Nginx, which is a web server that can also be used as a reverse proxy, load balancer. The application can be orchestrated with an orchestration system such as k8s for scalability, manageability, and monitoring.
