import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Setup the Face Mesh model
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh contours
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
        # Display the resulting image
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        
        # Press 'q' on your keyboard to exit the loop and close the window
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Clean up and release the camera
cap.release()
cv2.destroyAllWindows()
