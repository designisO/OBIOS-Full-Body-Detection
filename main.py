import mediapipe as mp
import cv2

# components of drawing and model
mp_drawing = mp.solutions.drawing_utils # drawings
mp_holistic = mp.solutions.holistic #models

# Webcam
cap = cv2.VideoCapture(0)

# initiate model
with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence =0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        # recolor image on feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # make detection 
        results = holistic.process(image)
        # print(results.face_landmarks)

        # recolor image back to BGR for rendering on feed
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


        # Making detections on cam below

        # drawing face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

        # drawing pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # drawing left hand landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # drawing right hand landmarks
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)



        cv2.imshow('OBIOS Full Body Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()