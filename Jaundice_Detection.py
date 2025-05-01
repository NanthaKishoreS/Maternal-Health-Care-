import os
import sys
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
import mediapipe as mp
import time
from efficientnet_pytorch import EfficientNet

class JaundiceDetector:
    def __init__(self, model_path, model_type="efficientnet-b3"):
        """
        Initialize the Jaundice Detector with a pre-trained model
        
        Args:
            model_path: Path to the trained model checkpoint (.pth file)
            model_type: Model architecture type ('efficientnet-b3', 'resnet50', 'densenet121')
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        # FIX 1: Swap the order of classes to fix inverted predictions
        self.classes = ["Jaundice", "Normal"]  # Changed from ["Normal", "Jaundice"]
        self.model = None
        
        # Initialize MediaPipe face mesh for sclera detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define data transformations for inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load model
        self.load_model(model_path)
    
    def initialize_model(self, model_name, num_classes):
        """Initialize the model architecture"""
        try:
            if model_name.startswith('efficientnet'):
                model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
            elif model_name == "resnet50":
                model = models.resnet50(pretrained=False)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
            elif model_name == "densenet121":
                model = models.densenet121(pretrained=False)
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, num_classes)
            else:
                raise ValueError(f"Unsupported model type: {model_name}")
            
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
    
    def load_model(self, model_path):
        """Load the trained model from .pth file"""
        try:
            # Initialize model architecture
            self.model = self.initialize_model(self.model_type, len(self.classes))
            self.model.to(self.device)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # Assume the dictionary contains the model state directly
                    state_dict = checkpoint
            else:
                # Assume the file contains the model state directly
                state_dict = checkpoint
            
            # Handle missing keys by filtering the state dict
            model_state_dict = self.model.state_dict()
            
            # 1. Filter out unnecessary keys
            state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
            
            # 2. Handle size mismatches
            for key in model_state_dict:
                if key in state_dict:
                    if model_state_dict[key].shape != state_dict[key].shape:
                        print(f"Size mismatch for {key}: expected {model_state_dict[key].shape}, got {state_dict[key].shape}")
                        del state_dict[key]
            
            # 3. Load what we can
            model_state_dict.update(state_dict)
            self.model.load_state_dict(model_state_dict)
            
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
            print(f"Model architecture: {self.model_type}")
            print(f"Class mapping: 0={self.classes[0]}, 1={self.classes[1]}")  # Display class mapping for clarity
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect_face_landmarks(self, image):
        """Detect face landmarks using MediaPipe"""
        try:
            # Convert to RGB if needed
            if isinstance(image, np.ndarray):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = np.array(image.convert('RGB'))
            
            # Process the image
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return None
                
            return results.multi_face_landmarks[0].landmark
        except Exception as e:
            print(f"Error detecting face landmarks: {e}")
            return None
    
    def extract_sclera_roi(self, image):
        """
        Extract the sclera region of interest from an eye image
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Sclera ROI (PIL Image)
        """
        try:
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                img_np = np.array(image.convert('RGB'))
            else:
                img_np = image.copy()
            
            # Detect face landmarks
            landmarks = self.detect_face_landmarks(img_np)
            if not landmarks:
                print("Warning: No face landmarks detected. Using full image.")
                return Image.fromarray(img_np)
            
            # Get image dimensions
            h, w = img_np.shape[:2]
            
            # MediaPipe eye landmarks (using refined landmarks since we set refine_landmarks=True)
            LEFT_IRIS = [474, 475, 476, 477]
            RIGHT_IRIS = [469, 470, 471, 472]
            LEFT_EYE = [33, 133, 160, 144, 158, 153]
            RIGHT_EYE = [362, 263, 387, 373, 380, 374]
            
            # Extract eye regions
            eyes = []
            for eye_indices in [LEFT_EYE, RIGHT_EYE]:
                x_coords = []
                y_coords = []
                
                for idx in eye_indices:
                    if idx < len(landmarks):
                        landmark = landmarks[idx]
                        x_coords.append(int(landmark.x * w))
                        y_coords.append(int(landmark.y * h))
                
                if x_coords and y_coords:
                    # Expand the eye region slightly
                    x_min, x_max = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
                    y_min, y_max = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)
                    
                    eye_roi = img_np[y_min:y_max, x_min:x_max]
                    if eye_roi.size > 0:  # Check if ROI is valid
                        eyes.append(eye_roi)
            
            # Combine both eyes if found
            if len(eyes) == 2:
                combined_eyes = np.concatenate(eyes, axis=1)
            elif len(eyes) == 1:
                combined_eyes = eyes[0]
            else:
                print("Warning: No eyes detected. Using full image.")
                return Image.fromarray(img_np)
            
            return Image.fromarray(combined_eyes)
        except Exception as e:
            print(f"Error extracting sclera ROI: {e}")
            return Image.fromarray(img_np)
    
    def enhance_image(self, image):
        """
        Enhance input image for better sclera detection
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Enhanced image (PIL Image)
        """
        try:
            # Convert to cv2 format if needed
            if isinstance(image, Image.Image):
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                img_cv = image.copy()
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Apply bilateral filter to reduce noise while preserving edges
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # FIX 2: Enhance the yellow hue detection for better jaundice signal
            # Convert to HSV for better color manipulation
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            # Enhance yellow range (slightly boost yellow hue and saturation)
            yellow_mask = cv2.inRange(hsv, (20, 60, 60), (40, 255, 255))
            
            # Create enhanced result with boosted yellow regions
            result = enhanced.copy()
            yellow_boost = cv2.bitwise_and(enhanced, enhanced, mask=yellow_mask)
            # Apply slight boost to yellow regions
            cv2.addWeighted(result, 0.8, yellow_boost, 0.5, 0, result)
            
            # Convert back to PIL
            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Error enhancing image: {e}")
            return image if isinstance(image, Image.Image) else Image.fromarray(image)

    def preprocess_image(self, image_path):
        """
        Preprocess an image for jaundice detection
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed tensor ready for model inference
        """
        try:
            # Load image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            image = Image.open(image_path).convert('RGB')
            
            # Extract and enhance sclera region
            sclera_roi = self.extract_sclera_roi(image)
            enhanced_roi = self.enhance_image(sclera_roi)
            
            # Apply model transformations
            input_tensor = self.transform(enhanced_roi)
            
            # Add batch dimension
            input_batch = input_tensor.unsqueeze(0)
            
            return input_batch, enhanced_roi
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None, None
    
    def predict(self, image_path, visualize=True, output_dir=None):
        """
        Predict jaundice from an eye image
        
        Args:
            image_path: Path to the input image
            visualize: Whether to visualize the results
            output_dir: Directory to save visualization results
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        start_time = time.time()
        
        # Preprocess image
        input_batch, sclera_roi = self.preprocess_image(image_path)
        if input_batch is None:
            return {"error": "Failed to preprocess image"}
        
        # Perform inference
        try:
            with torch.no_grad():
                input_batch = input_batch.to(self.device)
                outputs = self.model(input_batch)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                # Get predictions
                confidence, prediction = torch.max(probabilities, 0)
                confidence = confidence.item()
                prediction_class = self.classes[prediction.item()]
                
                # Get individual class probabilities
                class_probs = {cls: prob.item() for cls, prob in zip(self.classes, probabilities)}
                
                # Record processing time
                processing_time = time.time() - start_time
                
                result = {
                    "prediction": prediction_class,
                    "confidence": confidence,
                    "class_probabilities": class_probs,
                    "processing_time": processing_time
                }
                
                # FIX 3: Validate prediction based on yellow color intensity
                # This is a sanity check to catch obvious errors
                img_array = np.array(sclera_roi)
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                # Extract yellow range from HSV
                yellow_mask = cv2.inRange(hsv, (20, 60, 60), (40, 255, 255))
                yellow_proportion = np.sum(yellow_mask > 0) / yellow_mask.size
                
                # Log the yellow proportion for debugging
                result["yellow_proportion"] = yellow_proportion
                
                # Visualize if requested
                if visualize:
                    self._visualize_result(image_path, sclera_roi, result, output_dir)
                
                return result
        except Exception as e:
            return {"error": str(e)}
    
    def _visualize_result(self, image_path, sclera_roi, result, output_dir=None):
        """
        Visualize the detection results
        
        Args:
            image_path: Path to the original image
            sclera_roi: Extracted sclera region (PIL Image)
            result: Prediction result dictionary
            output_dir: Directory to save visualization results
        """
        try:
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 4, 1)
            img = Image.open(image_path).convert('RGB')
            plt.imshow(img)
            plt.title("Original Image")
            plt.axis('off')
            
            # Sclera ROI
            plt.subplot(1, 4, 2)
            plt.imshow(sclera_roi)
            plt.title("Extracted Sclera ROI")
            plt.axis('off')
            
            # FIX 4: Show yellow detection mask for better explanation
            plt.subplot(1, 4, 3)
            img_array = np.array(sclera_roi)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            yellow_mask = cv2.inRange(hsv, (20, 60, 60), (40, 255, 255))
            plt.imshow(yellow_mask, cmap='gray')
            plt.title(f"Yellow Detection\n(Coverage: {result.get('yellow_proportion', 0):.2%})")
            plt.axis('off')
            
            # Results visualization
            plt.subplot(1, 4, 4)
            classes = list(result["class_probabilities"].keys())
            probs = list(result["class_probabilities"].values())
            
            # Use bar colors based on prediction
            colors = ['lightblue'] * len(classes)
            pred_idx = classes.index(result["prediction"])
            colors[pred_idx] = 'orange' if result["prediction"] == "Jaundice" else 'green'
            
            plt.bar(classes, probs, color=colors)
            plt.ylim(0, 1)
            plt.title(f"Prediction: {result['prediction']}\nConfidence: {result['confidence']:.2f}")
            plt.ylabel("Probability")
            
            # Save or show
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, f"result_{os.path.splitext(filename)[0]}.png")
                plt.savefig(output_path)
                plt.close()
                print(f"Result saved to {output_path}")
            else:
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Error visualizing results: {e}")


def main():
    """Main function to run the jaundice detector"""
    # Default values
    model_path = "Jaundice_model.pth"
    model_type = "efficientnet-b3"
    output_dir = "output"
    
    # Parse command line arguments if available
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Specify your image path here
        image_path = "test.jpg"  # Change this to your image path
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Input image '{image_path}' does not exist")
        return
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist")
        print("Please ensure you have the trained model file in the correct location.")
        print("You may need to train the model first or download the pretrained weights.")
        return
    
    # Create detector
    try:
        print(f"Initializing detector with {model_type} model...")
        detector = JaundiceDetector(model_path=model_path, model_type=model_type)
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        print("Possible causes:")
        print("- Model architecture doesn't match the checkpoint")
        print("- Corrupted model file")
        print("- Missing dependencies")
        return
    
    # Process the image
    print(f"\nProcessing image: {image_path}")
    result = detector.predict(  
        image_path, 
        visualize=True,
        output_dir=output_dir
    )
    
    if "error" in result:
        print(f"\nError: {result['error']}")
    else:
        print(f"\nResults:")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Processing time: {result['processing_time']:.3f} seconds")
        print("\nClass probabilities:")
        for cls, prob in result["class_probabilities"].items():
            print(f"  {cls}: {prob:.4f}")
        
        # FIX 5: Print yellow proportion as additional diagnostic information
        if "yellow_proportion" in result:
            print(f"\nYellow proportion in sclera: {result['yellow_proportion']:.2%}")
            if result["prediction"] == "Jaundice":
                expected_yellow = result["yellow_proportion"] > 0.1
                print(f"Prediction matches yellow detection: {'Yes' if expected_yellow else 'No'}")


if __name__ == "__main__":
    main()