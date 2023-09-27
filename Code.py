import cv2

# Load the pre-trained Haar Cascade classifier for car detection
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Load the pre-trained HOG SVM for pedestrian detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open the input video file
input_video = cv2.VideoCapture('input_video.mp4')

# Get the codec information of the input video
fourcc = int(input_video.get(cv2.CAP_PROP_FOURCC))
fps = int(input_video.get(cv2.CAP_PROP_FPS))
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create an output video writer
output_video = cv2.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = input_video.read()
    if not ret:
        break

    # Convert the frame to grayscale for car detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect pedestrians in the frame
    pedestrians, _ = hog.detectMultiScale(frame)

    # Draw rectangles around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Draw rectangles around detected pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Write the frame to the output video
    output_video.write(frame)

# Release video objects
input_video.release()
output_video.release()

cv2.destroyAllWindows()