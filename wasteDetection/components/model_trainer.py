import os,sys
import yaml
from wasteDetection.utils.main_utils import read_yaml_file
from wasteDetection.logger import logging
from wasteDetection.exception import AppException
from wasteDetection.entity.config_entity import ModelTrainerConfig
from wasteDetection.entity.artifacts_entity import ModelTrainerArtifact



class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config



    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")
            import zipfile
            import shutil
            
            # Extract the zip file
            with zipfile.ZipFile("data.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Remove the zip file
            os.remove("data.zip")

            with open("data.yaml", 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(model_config_file_name)

            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")

            config['nc'] = int(num_classes)


            with open(f'yolov5/models/custom_{model_config_file_name}.yaml', 'w') as f:
                yaml.dump(config, f)

            os.system(f"cd yolov5 && python train.py --img 416 --batch {self.model_trainer_config.batch_size} --epochs {self.model_trainer_config.no_epochs} --data ../data.yaml --cfg ./models/custom_yolov5s.yaml --weights {self.model_trainer_config.weight_name} --name yolov5s_results  --cache --workers 0 --exist-ok")
            
            # Find the actual results directory (could be yolov5s_results, yolov5s_results2, etc.)
            import glob
            result_dirs = glob.glob("yolov5/runs/train/yolov5s_results*/weights/best.pt")
            if not result_dirs:
                raise FileNotFoundError("Training completed but best.pt not found in any results directory")
            
            # Use the most recent results directory
            best_pt_path = sorted(result_dirs)[-1]
            
            # Copy best.pt to yolov5 folder
            shutil.copy(best_pt_path, "yolov5/")
            
            # Copy to model trainer directory
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            shutil.copy(best_pt_path, self.model_trainer_config.model_trainer_dir)
           
            # Cleanup - remove directories and files
            if os.path.exists("yolov5/runs"):
                shutil.rmtree("yolov5/runs")
            if os.path.exists("train"):
                shutil.rmtree("train")
            if os.path.exists("valid"):
                shutil.rmtree("valid")
            if os.path.exists("data.yaml"):
                os.remove("data.yaml")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov5/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact


        except Exception as e:
            raise AppException(e, sys)
