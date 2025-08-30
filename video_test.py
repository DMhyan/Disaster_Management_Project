import cv2
import os
from ultralytics import YOLO
# Define a function for testing the video
def test_video():
    MODEL_PATH = "runs/detect/train/weights/best.pt"
    DATASET_PATH = r"C:\College\Disaster Management Project\dataset\search-and-rescue-2"
    
    print("SAR Model Testing Suite")
    print("=" * 50)
    
    # Initialize tester
    tester = SARModelTester(MODEL_PATH, DATASET_PATH)
    # Get the current working directory
    current_dir = os.getcwd()

    # Construct the full path to the video file dynamically
    test_video_path = os.path.join(current_dir, "dataset", "search-and-rescue-2", "data_1.mp4")

    # Check if the video file exists
    if os.path.exists(test_video_path):
        # Open video using OpenCV
        cap = cv2.VideoCapture(test_video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
        else:
            # Loop through the video frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Pass the frame to the YOLO model for inference
                tester.test_single_image(frame)  # Assuming `tester.test_single_image` can handle frame input

                # Optional: Display the frame (if you want to see the video during processing)
                cv2.imshow("Video Frame", frame)

                # Optional: Exit video display with 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the video capture and close any OpenCV windows
            cap.release()
            cv2.destroyAllWindows()
    else:
        print(f"Error: The file at '{test_video_path}' does not exist!")

# Call the function to test the video
test_video()
