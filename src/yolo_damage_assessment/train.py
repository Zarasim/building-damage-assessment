import os
import yaml
import mlflow
import argparse
from ultralytics import YOLO
from data.data_preprocessor import DataPreprocessor
from data.data_loader import DataLoader
from utils.logging_config import setup_logger

# Set up argument parser
parser = argparse.ArgumentParser(description='Train YOLOv8 model for building damage assessment')
parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file')
parser.add_argument('--output', type=str, default='models/trained_models', help='Path to save the trained model and results')

args = parser.parse_args()

# Set up logging
logger = setup_logger('training.log')

def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_model(config_path: str, output_dir: str):
    """
    Train the YOLOv8 model using the specified configuration.

    Args:
        config_path (str): Path to the configuration file.
        output_dir (str): Path to save the trained model and results.

    Returns:
        None
    """
    try:
        logger.info("Starting model training")

        # Load configuration
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")

        mlflow.start_run()

        # Data preprocessing and augmentation
        preprocessor = DataPreprocessor(config)
        preprocessor.preprocess_images()
        preprocessor.augment_images()
        logger.info("Completed data preprocessing and augmentation")

        # Data loading and preparation
        data_loader = DataLoader(config)
        data_loader.prepare_yolo_dataset()
        data_yaml_path = data_loader.get_data_yaml_path()
        logger.info("Completed data loading and preparation")

        # Set up CUDA environment variable
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"
        logger.info("Set CUDA environment variable")

        # Initialize the model
        model = YOLO(config['model']['name'])
        logger.info(f"Initialized YOLO model: {config['model']['name']}")

        # Train the model
        logger.info("Starting model training")
        results = model.train(
            data=data_yaml_path,
            epochs=config['training']['epochs'],
            batch=config['training']['batch_size'],
            imgsz=config['model']['input_size'],
            save_dir=output_dir
        )

        # Log parameters and metrics
        mlflow.log_params({
            "epochs": config['training']['epochs'],
            "batch_size": config['training']['batch_size'],
            "learning_rate": config['training']['learning_rate'],
            "model_name": config['model']['name'],
            "num_augmented_images": config['augmentation']['num_augmented_images']
        })

        mlflow.log_metrics({
            "mAP50": results.results_dict['metrics/mAP50(B)'],
            "mAP50-95": results.results_dict['metrics/mAP50-95(B)']
        })

        # Save the model
        mlflow.pytorch.log_model(model, "yolov8_model")

        # Print and log training results
        logger.info(f"Training completed. Results saved to {output_dir}")
        logger.info(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
        logger.info(f"Best mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")

    except Exception as e:
        logger.exception(f"An error occurred during training: {str(e)}")
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    train_model(args.config, args.output)