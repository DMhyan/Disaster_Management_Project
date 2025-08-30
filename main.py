import cv2
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import logging
from datetime import datetime
from ultralytics import YOLO
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SARDatasetManager:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.classes = [
            "injured_motionless",
            "walking",
            "sitting", 
            "lying_down",
            "waving_gesturing",
            "running_moving"
        ]
        
    def create_data_yaml(self):
        data_yaml = {
            'path': str(self.dataset_path.absolute()).replace('\\', '/'),
            'train': 'train/images',
            'val': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        yaml_path = self.dataset_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"Created data.yaml at {yaml_path}")
        return yaml_path
    
    def analyze_dataset(self):
        stats = {
            'train': {'images': 0, 'labels': 0, 'classes': {i: 0 for i in range(len(self.classes))}},
            'test': {'images': 0, 'labels': 0, 'classes': {i: 0 for i in range(len(self.classes))}}
        }
        
        for split in ['train', 'test']:
            images_path = self.dataset_path / split / 'images'
            labels_path = self.dataset_path / split / 'labels'
            
            if images_path.exists():
                jpg_files = list(images_path.glob('*.jpg'))
                png_files = list(images_path.glob('*.png'))
                stats[split]['images'] = len(jpg_files) + len(png_files)
            
            if labels_path.exists():
                label_files = list(labels_path.glob('*.txt'))
                stats[split]['labels'] = len(label_files)
                
                for label_file in label_files:
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    parts = line.split()
                                    if len(parts) >= 5:
                                        class_id = int(parts[0])
                                        if 0 <= class_id < len(self.classes):
                                            stats[split]['classes'][class_id] += 1
                    except Exception as e:
                        logger.warning(f"Error reading {label_file}: {e}")
        
        return stats

class SARYOLOTrainer:
    def __init__(self, model_version='yolov8n.pt'):
        self.model_version = model_version
        self.model = None
        self.results = None
        
    def train_model(self, data_yaml_path, epochs=50, imgsz=640, batch=8):
        try:
            self.model = YOLO(self.model_version)
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            
            train_args = {
                'data': str(data_yaml_path),
                'epochs': epochs,
                'imgsz': imgsz,
                'batch': batch,
                'device': device,
                'patience': 10,
                'save_period': 10,
                'cache': False,
                'workers': 0,
                'project': 'runs/detect',
                'name': 'train'
            }
            
            logger.info(f"Starting training with {epochs} epochs on {device}...")
            self.results = self.model.train(**train_args)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def validate_model(self, data_yaml_path):
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        val_results = self.model.val(data=str(data_yaml_path), device=device)
        return val_results

class SARInferenceEngine:
    def __init__(self, model_path, class_names=None):
        try:
            self.model = YOLO(model_path)
            self.class_names = class_names or [
                "injured_motionless",
                "walking", 
                "sitting",
                "lying_down",
                "waving_gesturing",
                "running_moving"
            ]
            self.colors = self._generate_colors()
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _generate_colors(self):
        np.random.seed(42)
        colors = []
        for i in range(len(self.class_names)):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)
        return colors
    
    def detect_image(self, image_path, conf_threshold=0.25, save_result=True):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            results = self.model(str(image_path), conf=conf_threshold, device=device, verbose=False)
            
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"Could not load image: {image_path}")
                return [], None
            
            detections = []
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        if cls < len(self.class_names):
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class': cls,
                                'class_name': self.class_names[cls]
                            }
                            detections.append(detection)
                            
                            color = self.colors[cls]
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            
                            label = f"{self.class_names[cls]}: {conf:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(img, (int(x1), int(y1) - label_size[1] - 10), 
                                         (int(x1) + label_size[0], int(y1)), color, -1)
                            cv2.putText(img, label, (int(x1), int(y1) - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if save_result:
                output_path = f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(output_path, img)
                logger.info(f"Detection result saved to {output_path}")
            
            return detections, img
            
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            return [], None
    
    def detect_dataset(self, dataset_path, split="test", conf_threshold=0.25, save_results=True):
        images_path = Path(dataset_path) / split / "images"
        if not images_path.exists():
            logger.error(f"Images path does not exist: {images_path}")
            return []
        
        results_all = []
        out_dir = Path(f"{split}_detections")
        
        if save_results:
            out_dir.mkdir(exist_ok=True)
        
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        logger.info(f"Processing {len(image_files)} images from {split} dataset")
        
        for i, img_path in enumerate(image_files):
            detections, result_img = self.detect_image(img_path, conf_threshold, save_result=False)
            results_all.extend(detections)
            
            if save_results and result_img is not None:
                cv2.imwrite(str(out_dir / img_path.name), result_img)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_files)} images")
        
        logger.info(f"Completed processing. Total detections: {len(results_all)}")
        return results_all

def generate_sar_report(detections, output_file="sar_report.txt"):
    report = []
    report.append("=" * 50)
    report.append("SEARCH AND RESCUE DETECTION REPORT")
    report.append("=" * 50)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    class_counts = {}
    for det in detections:
        class_name = det['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    report.append("DETECTION SUMMARY:")
    report.append("-" * 20)
    total_people = len(detections)
    report.append(f"Total people detected: {total_people}")
    
    high_priority = class_counts.get('injured_motionless', 0)
    medium_priority = class_counts.get('lying_down', 0) + class_counts.get('sitting', 0)
    low_priority = class_counts.get('walking', 0) + class_counts.get('running_moving', 0)
    attention_needed = class_counts.get('waving_gesturing', 0)
    
    report.append(f"HIGH PRIORITY (Injured/Motionless): {high_priority}")
    report.append(f"MEDIUM PRIORITY (Lying/Sitting): {medium_priority}")
    report.append(f"ATTENTION NEEDED (Waving/Gesturing): {attention_needed}")
    report.append(f"LOW PRIORITY (Mobile): {low_priority}")
    report.append("")
    
    report.append("DETAILED BREAKDOWN:")
    report.append("-" * 20)
    for class_name, count in class_counts.items():
        report.append(f"{class_name}: {count}")
    
    report.append("")
    report.append("RECOMMENDATIONS:")
    report.append("-" * 15)
    
    if high_priority > 0:
        report.append(f"IMMEDIATE ACTION REQUIRED: {high_priority} injured/motionless person(s) detected")
    if attention_needed > 0:
        report.append(f"{attention_needed} person(s) actively signaling for help")
    if medium_priority > 0:
        report.append(f"{medium_priority} person(s) in potentially vulnerable positions")
    if low_priority > 0:
        report.append(f"{low_priority} mobile person(s) detected")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"SAR report saved to {output_file}")
    return report

def main():
    DATASET_PATH = r"C:\College\Disaster Management Project\dataset\search-and-rescue-2"
    MODEL_VERSION = "yolov8n.pt"
    
    print("AI-Based Disaster Management System")
    print("=" * 50)
    
    try:
        print("Setting up dataset...")
        dataset_manager = SARDatasetManager(DATASET_PATH)
        
        if not Path(DATASET_PATH).exists():
            logger.error(f"Dataset path does not exist: {DATASET_PATH}")
            return None
            
        data_yaml_path = dataset_manager.create_data_yaml()
        stats = dataset_manager.analyze_dataset()
        
        print("Dataset Statistics:")
        for split, data in stats.items():
            print(f"  {split}: {data['images']} images, {data['labels']} labels")
            print(f"    Class distribution: {data['classes']}")
        
        print("\nTraining YOLO model...")
        trainer = SARYOLOTrainer(MODEL_VERSION)
        results = trainer.train_model(
            data_yaml_path=data_yaml_path,
            epochs=50,
            imgsz=640,
            batch=8
        )
        
        trained_model_path = "runs/detect/train/weights/best.pt"
        
        if not Path(trained_model_path).exists():
            logger.error("Trained model not found. Training may have failed.")
            return None
        
        print("\nValidating model...")
        val_results = trainer.validate_model(data_yaml_path)
        print(f"Validation mAP50: {val_results.box.map50:.3f}")
        print(f"Validation mAP50-95: {val_results.box.map:.3f}")
        
        print("\nRunning inference on test dataset...")
        inference_engine = SARInferenceEngine(trained_model_path)
        detections = inference_engine.detect_dataset(DATASET_PATH, split="test")
        
        if detections:
            report = generate_sar_report(detections)
            print("\nSAR System completed successfully!")
            print("Check 'sar_report.txt' for detailed results")
            print(f"Total detections: {len(detections)}")
        else:
            print("No detections found.")
        
        return {
            'dataset_manager': dataset_manager,
            'trainer': trainer,
            'inference_engine': inference_engine,
            'detections': detections,
            'validation_results': val_results
        }
        
    except Exception as e:
        logger.error(f"System failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    sar_system = main()