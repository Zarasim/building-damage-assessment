# Building Damage Assessment

This project uses YOLOv8 for building footprint detection and damage assessment from satellite images.

## Project Structure

```
building_damage_assessment/
├── config/
│   └── config.yaml
├── utils/
│   └── logging_config.py
├── train_yolo.py
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
   python train_yolo.py
   ```

   You can also specify custom paths for the configuration file and output directory:
   ```
   python train_yolo.py --config path/to/your/config.yaml --output path/to/output/directory
   ```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
