# Building Damage Assessment

This project uses YOLOv8 for building footprint detection and damage assessment from satellite images.

## Project Structure

```
building_damage_assessment/
├── src/
│   ├── data/
│   │   ├── data_preprocessor.py
│   │   └── data_loader.py
│   ├── models/
│   │   └── yolov8_model.py
│   ├── utils/
│   │   └── logging_config.py
│   └── train.py
├── tests/
│   ├── test_data_preprocessor.py
│   └── test_data_loader.py
├── config/
│   └── config.yaml
├── requirements.txt
└── README.md
```

## Installation

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

## Usage

1. Update the `config/config.yaml` file with your specific settings.

2. Run the training script:
   ```
   python -m src.train
   ```

   You can also specify custom paths for the configuration file and output directory:
   ```
   python -m src.train --config path/to/your/config.yaml --output path/to/output/directory
   ```

## Running Tests

To run the tests:

```
pytest tests/
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.