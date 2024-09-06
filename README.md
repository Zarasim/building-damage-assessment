# Building Damage Assessment

This project uses YOLOv8 for building footprint detection and damage assessment from satellite images.

## Project Structure

```
building_damage_assessment/
├── config/
│   └── config.yaml
├── src/
│   └── yolo_damage_assessment/
│       ├── train_yolo.py
│       └── utils/
│           └── logging_config.py
├── tests/
│   └── test_train_yolo.py
├── requirements.txt
├── setup.py
└── README.md
```

## Local Installation

1. Clone the repository:
    ```
    git clone https://github.com/Zarasim/building-damage-assessment.git
    cd building-damage-assessment
    ```

2. Create a virtual environment and activate it:
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Local Usage

1. Update the `config/config.yaml` file with your specific settings.

2. Run the training script:
    ```
    python src/yolo_damage_assessment/train_yolo.py
    ```

    You can also specify custom paths for the configuration file and output directory:
    ```
    python src/yolo_damage_assessment/train_yolo.py --config path/to/your/config.yaml --output path/to/output/directory
    ```

## Training on Google Colab

To train the model on Google Colab using a dataset stored on Google Drive:

1. Upload your dataset to Google Drive.

2. Create a new Colab notebook.

3. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. Clone the repository:
   ```
   !git clone https://github.com/Zarasim/building-damage-assessment.git
   %cd building-damage-assessment
   ```

5. Install the required packages:
   ```
   !pip install -r requirements.txt
   ```

6. Update the `config/config.yaml` file to point to your dataset in Google Drive:
   ```python
   import yaml

   with open('config/config.yaml', 'r') as file:
       config = yaml.safe_load(file)

   config['data']['yaml_path'] = '/content/drive/MyDrive/path/to/your/data.yaml'
   config['data']['test_image'] = '/content/drive/MyDrive/path/to/your/test_image.jpg'

   with open('config/config.yaml', 'w') as file:
       yaml.dump(config, file)
   ```

7. Run the training script:
   ```
   !python src/yolo_damage_assessment/train_yolo.py
   ```

   Or with custom paths:
   ```
   !python src/yolo_damage_assessment/train_yolo.py --config /content/drive/MyDrive/path/to/your/config.yaml --output /content/drive/MyDrive/path/to/output/directory
   ```

8. The trained model and results will be saved to the specified output directory in your Google Drive.

## Testing

To run the tests, use pytest:

```
pytest tests/
```

## License

This project is licensed under the MIT License.
