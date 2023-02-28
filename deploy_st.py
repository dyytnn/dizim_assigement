import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from utils.model import build_model
import dlib
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import time
import tempfile
import os

# Load the trained model
model = build_model(input_shape = (128, 128, 3))
model.load_weights('./weight/transfer_inception_resnet.h5')
detector = dlib.get_frontal_face_detector()
# Define the class labels
class_names = ["Fake", "Real"]


# Define the Streamlit app
def app():
    st.title("Deepfake Detector")

    # Upload a video
    uploaded_file = st.file_uploader("Choose a video", type=["mp4"])

    # If a video is uploaded
    if uploaded_file is not None:
        # f = st.file_uploader("Upload file")
        # vid = uploaded_file.name
        # with open(os.path.join("./videos/", vid), "wb") as f:
        #     f.write(uploaded_file.getbuffer())
        
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT); 
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter("./videos/output_video.mp4", fourcc, frame_rate - 10, (int(w),int(h)))
        
        frame_rate = 60
        prev = 0
        
        while True:
            # Read a frame from the video
            time_elapsed = time.time() - prev
            ret, frame = cap.read()
            # If the frame was read successfully
            if ret != True:
                break
            if time_elapsed > 1./frame_rate:
                face_rects, scores, idx = detector.run(frame, 0)
                for i, d in enumerate(face_rects):
                    x1 = d.left()
                    y1 = d.top()
                    x2 = d.right()
                    y2 = d.bottom()
                    crop_img = frame[y1:y2, x1:x2]
                    data = img_to_array(cv2.resize(crop_img, (128, 128))).flatten() / 255.0
                    data = data.reshape(-1, 128, 128, 3)
                    predict_x=model.predict(data) 
                    classes_x=np.argmax(predict_x,axis=1)
                    predicted_class = class_names[np.argmax(predict_x)]
                    cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (250, 18, 18), 4)
                    out.write(frame)
                prev = time.time()
        cap.release()
        out.release()
        
        video_file = open('./videos/output_video.mp4', 'rb') 
        video_bytes = video_file.read()
        st.video(video_bytes) 
        video_file.close()
        os.remove('./videos/output_video.mp4')
        
    cv2.destroyAllWindows()      

# Run the app
if __name__ == "__main__":
    app()

