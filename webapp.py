#imports 
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
#Add an expander to the app 

st.header("Biceps Curl Counter App")
with st.expander("About the App"):
        st.write("""
         This app is created using Streamlit and Mediapipe to count the number of biceps curls.
        """)    
#Adding a sidebar to the app
st.sidebar.title("Count your curls!")

run = st.sidebar.checkbox('Check to run CURL COUNTER and uncheck to stop!')
st.sidebar.text("Please put your hands in proper \nposition for biceps curls and start \nthe counter!")
FRAME_WINDOW = st.image([])

# Calculate Angles

def calculate_angle(a,b,c):
    # Reduce 3D point to 2D
    a = np.array([a.x, a.y])#, a.z])    
    b = np.array([b.x, b.y])#, b.z])
    c = np.array([c.x, c.y])#, c.z])  

    ab = np.subtract(a, b)
    bc = np.subtract(b, c)
    
    # A.B = |A||B|cos(x) 
    theta = np.arccos(np.dot(ab, bc) / np.multiply(np.linalg.norm(ab), np.linalg.norm(bc)))     
    # Convert to degrees
    theta = 180 - 180 * theta / 3.14   
    return np.round(theta, 2)


def curlCOunter():
    # Connecting Keypoints Visuals
    mp_drawing = mp.solutions.drawing_utils     

    # Keypoint detection model
    mp_pose = mp.solutions.pose     

    # Flag which stores hand position(Either UP or DOWN)
    left_flag = None     
    right_flag = None

    # Storage for count of bicep curls
    right_count = 0
    left_count = 0       

    cap = cv2.VideoCapture(0)
    # Landmark detection model instance
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) 
    while cap.isOpened():
        _, frame = cap.read()

         # Convert BGR frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     
        image.flags.writeable = False
        
        # Make Detections
        # Get landmarks of the object in frame from the model
        results = pose.process(image)   

        # Back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      

        try:
            # Extract Landmarks
            landmarks = results.pose_landmarks.landmark

            # Get coordinates of left part
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

            # Get coordinates of right part
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Calculate and get angle
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)      
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Visualize angle
            cv2.putText(image,\
                    str(left_angle), \
                        tuple(np.multiply([left_elbow.x, left_elbow.y], [640,480]).astype(int)),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
            cv2.putText(image,\
                    str(right_angle), \
                        tuple(np.multiply([right_elbow.x, right_elbow.y], [640,480]).astype(int)),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
        
            # Counter 
            if left_angle > 160:
                left_flag = 'down'
            if left_angle < 50 and left_flag=='down':
                left_count += 1
                left_flag = 'up'

            if right_angle > 160:
                right_flag = 'down'
            if right_angle < 50 and right_flag=='down':
                right_count += 1
                right_flag = 'up'
            
        except:
            pass

        # Setup Status Box
        cv2.rectangle(image, (0,0), (1024,73), (10,10,10), -1)
        cv2.putText(image, 'Left=' + str(left_count) + '    Right=' + str(right_count),
                          (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.imshow('MediaPipe feed', image)

        FRAME_WINDOW.image(image)

        # Esc for quiting the app
        k = cv2.waitKey(30) & 0xff  
        if k==27:
            break
        elif k==ord('r'):       
            # Reset the counter on pressing 'r' on the Keyboard
            left_count = 0
            right_count = 0

    cap.release()
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    while run:
        curlCOunter()
    else:
        st.write('Stopped')   
