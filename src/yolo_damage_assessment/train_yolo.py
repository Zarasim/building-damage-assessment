import os
import yaml
import argparse
from ultralytics import YOLO
import torch
from pathlib import Path
from yolo_damage_assessment.utils.logging_config import setup_logger

# Set up logging
logger = setup_logger('training.log')


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def train_model(config_path: str, output_dir: str):
    """Train the YOLOv8 model using the specified configuration."""
    try:
        logger.info("Starting model training")

        # Load configuration
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")

        # Set up CUDA environment variable
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"
        logger.info("Set CUDA environment variable")

        # Check CUDA availability
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

        # Initialize the model
        model = YOLO(config['model']['name'])
        logger.info(f"Initialized YOLO model: {config['model']['name']}")

        # Train the model
        logger.info("Starting model training")
        results = model.train(
            data=config['data']['yaml_path'],
            epochs=config['training']['epochs'],
            imgsz=config['model']['input_size'],
            batch=config['training']['batch_size'],
            name=config['training']['run_name'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Validate the model
        val_results = model.val()
        logger.info(f"Validation results: {val_results}")

        # Perform inference on a test image
        test_image_path = config['data']['test_image']
        test_results = model.predict(test_image_path, save=True, conf=0.5)
        logger.info(f"Test results: {test_results}")

        # Save the trained model
        output_path = Path(output_dir) / f"{config['training']['run_name']}_best.pt"
        torch.save(model.state_dict(), output_path)
        logger.info(f"Saved best model to {output_path}")

        # Print and log training results
        logger.info(f"Training completed. Results saved to {output_dir}")
        logger.info(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
        logger.info(f"Best mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")

    except Exception as e:
        logger.exception(f"An error occurred during training: {str(e)}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for building damage assessment')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file')
    parser.add_argument('--output', type=str, default='models/trained_models',
                        help='Path to save the trained model and results')

    args = parser.parse_args()

    train_model(args.config, args.output)


if __name__ == "__main__":
    main()
