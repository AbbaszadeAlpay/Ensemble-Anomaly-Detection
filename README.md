# Ensemble Anomaly Detection Models
This project is centered around anomaly detection, employing an ensemble technique utilizing the PyOD library. The primary goal is to identify anomalies within datasets by leveraging a combined approach of five distinct models provided by PyOD.

Anomaly detection is a critical aspect of data analysis, allowing for the identification of data points that deviate significantly from the norm. In this project, an ensemble methodology is adopted, where the collective decision of multiple models determines whether a data point is an anomaly or normal.

The ensemble technique operates as follows: if three or more models out of the five flag a data point as anomalous, the system labels it as an anomaly. Conversely, if fewer than three models indicate an anomaly, the data point is classified as normal.

The PyOD library offers a diverse set of anomaly detection models, each with its strengths and characteristics. By combining these models into an ensemble, this project aims to enhance the detection accuracy and robustness across various types of datasets.

The strength of this ensemble-based approach lies in its ability to provide a more comprehensive and nuanced view of anomalies within the dataset. Leveraging multiple models helps capture diverse aspects of anomalies and normal data points, leading to a more informed decision-making process.

The project's ultimate objective is to provide a reliable and adaptable anomaly detection system that can be applied across different domains and datasets, improving anomaly detection capabilities and aiding in the identification of irregularities within complex data structures.


## Installation

#### Poetry Installation: 
Poetry is used for managing project dependencies. Once Python is installed, install Poetry via the terminal or command prompt with the following command

```bash
curl -sSL https://install.python-poetry.org | python -

```
#### Downloading or Cloning the Project:
Download or clone the project from GitHub:    

```bash
https://github.com/AbbaszadeAlpay/Ensemble-Anomaly-Dedection.git

```
#### Installing Project Dependencies: 
Navigate to the project folder and install the dependencies:
```bash
cd <project_folder>
poetry install
```


## Usage/Examples
To utilize the anomaly detection functionality using this code snippet, follow these steps:
#### 1. Import Necessary Modules:
Import the required modules for dataset I/O operations and model training:
```bash
from src.dataset_io import DatasetIO
from src.model_training import AnomalyDetection
```
#### 2. Read Dataset:
Read the dataset using DatasetIO from the specified source (in this case, a CSV file named transactions.csv within the data_source directory):
```bash
df = DatasetIO(data_path).read_data()
```
#### 3. Initialize Anomaly Detection Model:
Initialize the anomaly detection model (AnomalyDetection) with the dataset (df) and specify the contamination level (in this case, 0.05):
```bash
anomaly_model = AnomalyDetection(df, contamination=0.05)
```
#### 4. Fit the model:
Train or fit the anomaly detection model on the provided dataset:
```bash
anomaly_model.fit()
```
By executing these steps sequentially, you will import the necessary modules, read your dataset, initialize an anomaly detection model, and train the model on your data. Adjust the dataset path and other parameters as per your project setup and requirements.

## API Usage
```bash
uvicorn src.model_training.api:app
```
```bash
http://127.0.0.1:8000/docs#/default/predict_predict_post
```

## Deployment

To deploy this project run

```bash
docker build -t anomalydedection . 
docker run -d -p 8041:8041 anomalydedection
```

## Authors

- [AbbaszadeAlpay](https://github.com/AbbaszadeAlpay)


## Acknowledgements

- [Frightera](https://github.com/Frightera)

## License

[MIT](https://choosealicense.com/licenses/mit/)









