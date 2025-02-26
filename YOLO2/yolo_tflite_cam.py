import cv2
import numpy as np
import tensorflow.lite as tflite

# Load the TFLite model
model_path = 'content/YOLOv11_det_640/yolov11_det_640/weights/best_saved_model/best_float16.tflite'  # Replace with your TFLite model path
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize, normalize, etc.)
    input_shape = input_details[0]['shape'][1:3]  # Get expected input shape (height, width)
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[0]))  # Resize to model input size
    normalized_frame = resized_frame / 255.0  # Normalize pixel values (if required by your model)
    input_data = np.expand_dims(normalized_frame, axis=0).astype(np.float32)  # Add batch dimension

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = output_data*100

    # Process output (e.g., draw bounding boxes, labels, etc.)
    # Assuming output_data contains detection results (e.g., bounding boxes, scores, classes)
    # You need to parse the output according to your model's output format

    # Example: Draw bounding boxes (adjust according to your model's output format)
    for detection in output_data[0]:  # Adjust based on your model's output structure
        score = detection[4]  # Confidence score
        if score > 0.5:  # Threshold for detection
            x_center, y_center, width, height = detection[:4]  # Bounding box coordinates in (x_center, y_center, width, height) format

            # Convert (x_center, y_center, width, height) to (xmin, ymin, xmax, ymax)
            xmin = int((x_center - width / 2) * frame.shape[1])
            ymin = int((y_center - height / 2) * frame.shape[0])
            xmax = int((x_center + width / 2) * frame.shape[1])
            ymax = int((y_center + height / 2) * frame.shape[0])

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'Pattern: {score:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Inference', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()