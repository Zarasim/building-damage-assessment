from setuptools import setup, find_packages

setup(
    name="building_damage_assessment",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "ultralytics==8.0.221",
        "opencv-python==4.8.1.78",
        "numpy==1.26.2",
        "PyYAML==6.0.1",
        "torch>=2.1.1",
        "torchvision==0.16.1",
        "mlflow==2.15.1"
    ],
    author="Simone Appella",
    author_email="simone.appella@gmail.com",
    description="A project for building damage assessment using YOLOv8",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Zarasim/building-damage-assessment",
)
