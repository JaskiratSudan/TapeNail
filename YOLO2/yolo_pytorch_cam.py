import cv2
import torch
import numpy as np

# Load the TorchScript model
model_path = 'content/YOLOv11_det_640/yolov11_det_640/weights/best.torchscript'  # Replace with your TorchScript model path
model = torch.jit.load(model_path)
model.eval()  # Set the model to evaluation mode

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Define model input size (replace with your model's expected input size)
input_size = (640, 640)  # YOLOv8 typically uses 640x640

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, input_size)  # Resize to model input size
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]
    input_tensor = torch.from_numpy(normalized_frame).permute(2, 0, 1).float()  # HWC to CHW
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Ensure the input tensor is on the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Run inference
    with torch.no_grad():  # Disable gradient calculation
        output = model(input_tensor)

    # Process output (YOLOv8 output format)
    # Assuming output is a list of tensors, where each tensor corresponds to a detection
    for detection in output:
        detection = detection.cpu().numpy()  # Convert to numpy array
        for *xyxy, conf, cls in detection:
            if conf > 0.5:  # Confidence threshold
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, xyxy)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Pattern: {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Inference', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()