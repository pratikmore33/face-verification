from fastapi import UploadFile,FastAPI,File
import tempfile
import uvicorn
import cv2
import numpy as np 
from deepface import DeepFace
import dlib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",'http://localhost:5000','http://localhost:8080'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def welcome():
    return {'message':'server check done'}

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks (1).dat')


# lets define function for extract face from image
def extract_img(image_path):
    original_image = image_path
    face_objs = DeepFace.extract_faces(img_path = image_path)
    # print(type(face_objs[-1]))
    # print(face_objs[-1]['facial_area'])
    for face_obj in face_objs:
       print(face_obj["facial_area"])
       x, y, w, h = map(int, face_obj["facial_area"].values())
       cropped_face = original_image[y:y+h, x:x+w]
       resize_image = cv2.resize(cropped_face,(224,224))
      #  cv2_imshow(resize_image)
       return resize_image

# lets define function for align image     
def align_face(image):
    # Create a copy of the image to avoid altering the original
    image_copy = image.copy()

    # Convert the image to grayscale (required by dlib's detector)
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    # If no faces are found, return the input image
    if len(faces) == 0:

        return image

    # Loop over the detected faces
    for face in faces:
        # Get the face's bounding box coordinates
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        # Predict facial landmarks for each detected face
        landmarks = predictor(gray, face)

        # Convert landmarks to NumPy array for easier manipulation
        landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Draw facial landmarks on the image copy (optional)
        for (x, y) in landmarks_np:
            cv2.circle(image_copy, (x, y), 2, (0, 255, 0), -1)  # Green circles for landmarks

        # Perform alignment (you can use landmarks_np to apply transformations)
        # For example, use the eye landmarks to align the face (this is a simple example)
        left_eye_center = landmarks_np[36:42].mean(axis=0)
        right_eye_center = landmarks_np[42:48].mean(axis=0)

        # Calculate the angle between the eyes
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Calculate the center point between the eyes
        eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)

        # Set the scale factor for resizing (you can adjust this as needed)
        scale = 1.0

        # Perform the affine transformation
        M = cv2.getRotationMatrix2D(eye_center, angle, scale)
        aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        aligned_face = cv2.resize(aligned_face, (224, 224))

        # Display the aligned face (optional)
        # cv2_imshow(aligned_face)

        # Save the aligned face without landmarks
        cv2.imwrite('aligned_face.jpg', aligned_face)

        return aligned_face

    # Return the input image if face detection fails
    return image

# model = 'arcface_weights.h5'
def verifyFace(img_1, img_2, distance_metrix):
    img_1_aligned = align_face(extract_img(img_1))
    img_2_aligned = align_face(extract_img(img_2))

    if img_1_aligned is None or img_2_aligned is None:
        return False  # Or handle this case as needed, could be invalid data or no face detected

    result = DeepFace.verify(img1_path=img_1_aligned, img2_path=img_2_aligned, model_name='ArcFace',enforce_detection=False,distance_metric= distance_metrix)
    # print(result['distance'])
    return result

import base64
import io

def process_input_image(input_data):
    # Check if the input is a base64-encoded string
    if isinstance(input_data, str) and input_data.startswith('data:image'):
        header, encoded = input_data.split(",", 1)
        decoded = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    elif isinstance(input_data, UploadFile):
        content_type = input_data.content_type
        if "image" in content_type:
            content = input_data.file.read()
            nparr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
    return None
        

@app.post('/verify')
async  def face_verification(image_1:UploadFile = File(...),image_2:UploadFile = File(...)):
    img_1 = process_input_image(image_1)
    img_2 = process_input_image(image_2)

    if img_1 is None or img_2 is None:
        return {"error": "Invalid input data"}

    
   

    

    result = verifyFace(img_1, img_2, 'cosine')
        
    if result['distance'] < 0.62:
        return {'verified':True,'Distance':result['distance']}
    else:
        return {'verified':False,'Distance':result['distance']}
    
if __name__ == '__main__':
    uvicorn.run(app,port=5000)


           
    
