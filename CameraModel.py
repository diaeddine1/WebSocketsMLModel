import cv2
import base64
import datetime
import websockets
import asyncio
import json
from ultralytics import YOLO, solutions

# Initialize YOLO model
model = YOLO("yolo11x.pt")

# Access the camera instead of a video file
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera
assert cap.isOpened(), "Error accessing the camera"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define class names and initialize counter
cars = []
persons = []

#WebSocket server connection
async def send_to_websocket(detected_object):
    uri = "wss://websocket-server-o3sb.onrender.com"  # WebSocket server URI
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(detected_object))

# Init Object Counter
counter = solutions.ObjectCounter(
    show=False,
    region=[(50, 50), (50, 1100)],  # Define region for counting
    classes=[0, 2],  # Only detect classes 'person' and 'car'
)

# Dictionary to keep track of unique IDs to prevent recounting
unique_objects = set()

# Process video
async def process_video():
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("No frame captured from camera. Exiting.")
            break

        # Run YOLO detection with tracking
        results = model.track(frame, persist=True, agnostic_nms=True)

        # Process detected objects in each frame
        for result in results:
            detection_count = result.boxes.shape[0]
            
            for i in range(detection_count):
                # Get class ID and name
                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]

                # Check if the tracker ID exists
                if result.boxes.id is not None:
                    object_id = result.boxes.id[i].item()
                else:
                    # Fallback: Generate a unique ID based on class and bounding box
                    object_id = hash((cls, tuple(result.boxes.xyxy[i].cpu().numpy())))

                bounding_box = result.boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, bounding_box)

                confidence = float(result.boxes.conf[i].item())

                # Check if this object ID has already been counted
                if object_id not in unique_objects:
                    unique_objects.add(object_id)

                    # Encode frame as JPG image and then convert to base64
                    _, buffer = cv2.imencode('.jpg', frame)  # Encode frame as JPG image
                    frame_binary = buffer.tobytes()  # Convert to bytes

                    # Convert the byte array to base64 string
                    frame_base64 = base64.b64encode(frame_binary).decode('utf-8')

                    detected_object = {
                        'yolo_class_id': cls,  # Class ID (0 for person, 2 for car)
                        'yolo_id': object_id,  # Unique object ID
                        'frame_image': frame_base64,  # Base64-encoded image
                        'frame_time': datetime.datetime.now().isoformat(),
                        'confidence': confidence,
                        'position_X': (x1 + x2) / 2,  # Center of bounding box on X axis
                        'position_Y': (y1 + y2) / 2,  # Center of bounding box on Y axis
                    }

                    #Send detected object to WebSocket server
                    await send_to_websocket(detected_object)

                    # Append detected object to relevant list
                    if detected_object["yolo_class_id"] == 0:
                        persons.append(detected_object)
                    elif detected_object["yolo_class_id"] == 2:
                        cars.append(detected_object)

                # Draw bounding box and confidence on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Update counter in each frame (this part is for counting objects)
        frame = counter.count(frame)

        # Display the frame (optional)
        cv2.imshow("YOLO Output", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(len(cars))
    print(len(persons))

# Run video processing asynchronously
async def main():
    await process_video()

if __name__ == "__main__":
    asyncio.run(main())
