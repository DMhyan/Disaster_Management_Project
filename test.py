import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from collections import defaultdict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SARModelTester:
    def __init__(self, model_path, dataset_path):
        self.model = YOLO(model_path)
        self.dataset_path = Path(dataset_path)
        self.class_names = [
            "injured_motionless",
            "walking", 
            "sitting",
            "lying_down", 
            "waving_gesturing",
            "running_moving"
        ]
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                      (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
    def test_single_image(self, image_path, conf_threshold=0.25, save_result=True):
        """Test model on a single image and display results"""
        print(f"Testing image: {Path(image_path).name}")
        
        # Run detection
        results = self.model(str(image_path), conf=conf_threshold)
        
        # Load image for visualization
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not load image: {image_path}")
            return []
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw detections
        detections_info = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    if cls < len(self.class_names):
                        # Store detection info
                        detection_info = {
                            'class': self.class_names[cls],
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        }
                        detections_info.append(detection_info)
                        
                        # Draw bounding box
                        color = self.colors[cls % len(self.colors)]
                        cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                        
                        # Add label
                        label = f"{self.class_names[cls]}: {conf:.2f}"
                        cv2.putText(img_rgb, label, (int(x1), int(y1-10)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display results
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"Detection Results - {Path(image_path).name}")
        
        if save_result:
            output_path = f"test_result_{Path(image_path).stem}.jpg"
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            print(f"Result saved to: {output_path}")
        
        plt.show()
        
        # Print detection summary
        print(f"\nDetections found in {Path(image_path).name}:")
        print("-" * 50)
        if detections_info:
            for i, det in enumerate(detections_info, 1):
                print(f"{i}. {det['class']} - Confidence: {det['confidence']:.3f}")
        else:
            print("No detections found")
        
        return detections_info
    
    def test_multiple_images(self, image_folder, num_images=5, conf_threshold=0.25):
        """Test model on multiple random images"""
        image_folder = Path(image_folder)
        image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
        
        if len(image_files) == 0:
            print(f"No images found in {image_folder}")
            return
        
        # Select random images
        selected_images = np.random.choice(image_files, min(num_images, len(image_files)), replace=False)
        
        all_detections = []
        for img_path in selected_images:
            print(f"\n{'='*60}")
            detections = self.test_single_image(img_path, conf_threshold, save_result=True)
            all_detections.extend(detections)
        
        # Summary statistics
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"Total images tested: {len(selected_images)}")
        print(f"Total detections: {len(all_detections)}")
        
        if all_detections:
            class_counts = {}
            confidence_scores = []
            for det in all_detections:
                class_counts[det['class']] = class_counts.get(det['class'], 0) + 1
                confidence_scores.append(det['confidence'])
            
            print(f"Average confidence: {np.mean(confidence_scores):.3f}")
            print("Class distribution:")
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count}")
        
        return all_detections
    
    def evaluate_on_test_set(self):
        """Run official YOLO evaluation on test set"""
        print("Running official YOLO evaluation...")
        
        data_yaml_path = self.dataset_path / "data.yaml"
        if not data_yaml_path.exists():
            print(f"data.yaml not found at {data_yaml_path}")
            return None
        
        # Run validation on test set
        metrics = self.model.val(data=str(data_yaml_path), split="test")
        
        print(f"\nOFFICIAL YOLO METRICS:")
        print(f"{'='*40}")
        print(f"mAP@0.5: {metrics.box.map50:.3f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
        print(f"Precision: {metrics.box.mp:.3f}")
        print(f"Recall: {metrics.box.mr:.3f}")
        
        # Per-class metrics
        print(f"\nPer-class mAP@0.5:")
        for i, class_name in enumerate(self.class_names):
            if i < len(metrics.box.maps):
                print(f"  {class_name}: {metrics.box.maps[i]:.3f}")
        
        return metrics
    
    def analyze_detection_confidence(self, conf_threshold=0.25):
        """Analyze confidence distribution across test set"""
        test_images_path = self.dataset_path / "test" / "images"
        
        all_confidences = []
        class_confidences = {class_name: [] for class_name in self.class_names}
        
        print("Analyzing confidence distributions...")
        image_files = list(test_images_path.glob("*.jpg"))
        
        for i, img_path in enumerate(image_files):
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
                
            results = self.model(str(img_path), conf=conf_threshold, verbose=False)
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        
                        if cls < len(self.class_names):
                            all_confidences.append(conf)
                            class_confidences[self.class_names[cls]].append(conf)
        
        # Plot confidence distributions
        if all_confidences:
            plt.figure(figsize=(15, 10))
            
            # Overall confidence distribution
            plt.subplot(2, 2, 1)
            plt.hist(all_confidences, bins=30, alpha=0.7, color='blue')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Overall Confidence Distribution')
            plt.axvline(np.mean(all_confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_confidences):.3f}')
            plt.legend()
            
            # Per-class confidence
            plt.subplot(2, 2, 2)
            class_means = []
            class_labels = []
            for class_name, confidences in class_confidences.items():
                if confidences:
                    class_means.append(np.mean(confidences))
                    class_labels.append(class_name)
            
            plt.bar(range(len(class_means)), class_means, color=self.colors[:len(class_means)])
            plt.xlabel('Class')
            plt.ylabel('Mean Confidence')
            plt.title('Mean Confidence by Class')
            plt.xticks(range(len(class_labels)), class_labels, rotation=45)
            
            # Box plot of confidence by class
            plt.subplot(2, 1, 2)
            conf_data = [confidences for confidences in class_confidences.values() if confidences]
            conf_labels = [name for name, confidences in class_confidences.items() if confidences]
            
            if conf_data:
                plt.boxplot(conf_data, labels=conf_labels)
                plt.xlabel('Class')
                plt.ylabel('Confidence Score')
                plt.title('Confidence Distribution by Class')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig('confidence_analysis.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"\nConfidence Analysis Results:")
            print(f"Total detections analyzed: {len(all_confidences)}")
            print(f"Overall mean confidence: {np.mean(all_confidences):.3f}")
            print(f"Overall std confidence: {np.std(all_confidences):.3f}")
            
        return all_confidences, class_confidences
    
    def create_detection_report(self, output_file="model_test_report.txt"):
        """Create comprehensive testing report"""
        print("Generating comprehensive test report...")
        
        report = []
        report.append("=" * 60)
        report.append("SAR MODEL TESTING REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: {self.model.ckpt_path}")
        report.append(f"Dataset: {self.dataset_path}")
        report.append("")
        
        # Official metrics
        try:
            metrics = self.evaluate_on_test_set()
            if metrics:
                report.append("OFFICIAL YOLO METRICS:")
                report.append("-" * 30)
                report.append(f"mAP@0.5: {metrics.box.map50:.3f}")
                report.append(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
                report.append(f"Precision: {metrics.box.mp:.3f}")
                report.append(f"Recall: {metrics.box.mr:.3f}")
                report.append("")
        except Exception as e:
            report.append(f"Official evaluation failed: {e}")
            report.append("")
        
        # Confidence analysis
        try:
            all_conf, class_conf = self.analyze_detection_confidence()
            if all_conf:
                report.append("CONFIDENCE ANALYSIS:")
                report.append("-" * 30)
                report.append(f"Total detections: {len(all_conf)}")
                report.append(f"Mean confidence: {np.mean(all_conf):.3f}")
                report.append(f"Std confidence: {np.std(all_conf):.3f}")
                report.append("")
                
                report.append("Per-class mean confidence:")
                for class_name, confidences in class_conf.items():
                    if confidences:
                        report.append(f"  {class_name}: {np.mean(confidences):.3f} ({len(confidences)} detections)")
        except Exception as e:
            report.append(f"Confidence analysis failed: {e}")
            report.append("")
        
        # Save report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Test report saved to: {output_file}")
        return report

def main():
    # Configuration
    MODEL_PATH = "runs/detect/train/weights/best.pt"
    DATASET_PATH = r"C:\College\Disaster Management Project\dataset\search-and-rescue-2"
    
    print("SAR Model Testing Suite")
    print("=" * 50)
    
    # Initialize tester
    tester = SARModelTester(MODEL_PATH, DATASET_PATH)
    
    # Test options
    print("\nSelect testing option:")
    print("1. Test single image")
    print("2. Test multiple random images")
    print("3. Full evaluation on test set")
    print("4. Confidence analysis")
    print("5. Generate comprehensive report")
    print("6. Run all tests")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        # Test single image
        test_images_path = "C:\College\Disaster Management Project\dataset\search-and-rescue-2\data_1.mp4"
        image_files = list(test_images_path.glob("*.mp4"))
        if image_files:
            test_image = image_files[0]  # Take first image
            tester.test_single_image(test_image)
        else:
            print("No test images found!")
    
    elif choice == "2":
        # Test multiple images
        test_images_path = Path(DATASET_PATH) / "test" / "images"
        tester.test_multiple_images(test_images_path, num_images=5)
    
    elif choice == "3":
        # Full evaluation
        tester.evaluate_on_test_set()
    
    elif choice == "4":
        # Confidence analysis
        tester.analyze_detection_confidence()
    
    elif choice == "5":
        # Generate report
        tester.create_detection_report()
    
    elif choice == "6":
        # Run all tests
        test_images_path = Path(DATASET_PATH) / "test" / "images"
        print("\n" + "="*60)
        print("RUNNING ALL TESTS")
        print("="*60)
        
        print("\n1. Testing multiple images...")
        tester.test_multiple_images(test_images_path, num_images=3)
        
        print("\n2. Running official evaluation...")
        tester.evaluate_on_test_set()
        
        print("\n3. Analyzing confidence distributions...")
        tester.analyze_detection_confidence()
        
        print("\n4. Generating comprehensive report...")
        tester.create_detection_report()
        
        print("\nAll tests completed!")
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()