import streamlit as st
import cv2
import torch
from utils.hubconf import custom
import numpy as np
import tempfile
import time
from collections import Counter
import json
import pandas as pd
from model_utils import get_yolo, color_picker_fn, get_system_stat
# from ultralytics import YOLO
import io
import os

p_time = 0

def open_image_as_file(path):
    # 读取图像
    img = cv2.imread(path)
    if img is None:
        st.error(f"Error: Unable to read the image from path: {path}")
        return None
    # 将图像编码为字节
    success, img_encoded = cv2.imencode('.jpg', img)
    if not success:
        st.error(f"Error: Unable to encode the image to bytes")
        return None
    # 使用 io.BytesIO 模拟文件对象
    img_file = io.BytesIO(img_encoded.tobytes())
    img_file.name = path  # 设置文件名属性
    return img_file


st.sidebar.title('Settings')
# Choose the model
model_type = st.sidebar.selectbox(
    # 'Choose YOLO Model', ('YOLO Model', 'YOLOv8', 'YOLOv7')
    'Choose YOLO Model', ('YOLO Model', 'carnumber')
)

st.title(f'{model_type} Predictions')
sample_img = cv2.imread('logo.jpg')
FRAME_WINDOW = st.image(sample_img, channels='BGR')
cap = None

path_model_file = st.sidebar.text_input(
    f'path to {model_type} Model:',
    f'{model_type}.pt'
)
if st.sidebar.checkbox('Load Model'):
        
     # YOLOv7 Model
    if model_type == 'carnumber':
        # GPU
        gpu_option = st.sidebar.radio(
            'PU Options:', ('CPU', 'GPU'))

        if not torch.cuda.is_available():
            st.sidebar.warning('CUDA Not Available, So choose CPU', icon="⚠️")
        else:
            st.sidebar.success(
                'GPU is Available on this Device, Choose GPU for the best performance',
                icon="✅"
            )
        # Model
        if gpu_option == 'CPU':
            model = custom(path_or_model=path_model_file)
        if gpu_option == 'GPU':
            model = custom(path_or_model=path_model_file, gpu=True)

        # YOLOv8 Model
        # if model_type == 'YOLOv8':
        #     from ultralytics import YOLO
        #     model = YOLO(path_model_file)

    # Load Class names
    class_labels = model.names

    # Inference Mode
    options = st.sidebar.radio(
        'Options:', ('Image', 'Video','Webcam'), index=0)

    # Confidence
    confidence = st.sidebar.slider(
        'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

    # Draw thickness
    draw_thick = st.sidebar.slider(
        'Draw Thickness:', min_value=1,
        max_value=20, value=2
    )
        
    color_pick_list = []
    for i in range(len(class_labels)):
        classname = class_labels[i]
        color = color_picker_fn(classname, i)
        color_pick_list.append(color)

  # Image
    if options == 'Image':
        option1 = st.sidebar.selectbox(
             'you can select some image',
             ('default','image_1', 'image_2'))
        if option1 =='image_1':
            upload_img_file = open_image_as_file('image/1.jpg')
        elif option1 =='image_2':
            upload_img_file = open_image_as_file('image/2.jpg')
        else:
            upload_img_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])
        if upload_img_file is not None:
            pred = st.checkbox(f'Predict Using {model_type}')
            file_bytes = np.asarray(
                bytearray(upload_img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            FRAME_WINDOW.image(img, channels='BGR')
    #   # Image
    # if options == 'Image':
    #     upload_img_file = st.sidebar.file_uploader(
    #         'Upload Image', type=['jpg', 'jpeg', 'png'])
    #     if upload_img_file is not None:
    #         pred = st.checkbox(f'Predict Using {model_type}')
    #         file_bytes = np.asarray(
    #             bytearray(upload_img_file.read()), dtype=np.uint8)
    #         img = cv2.imdecode(file_bytes, 1)
    #         FRAME_WINDOW.image(img, channels='BGR')
            if pred:
                img, current_no_class = get_yolo(img, model_type, model, confidence, color_pick_list, class_labels, draw_thick)
                FRAME_WINDOW.image(img, channels='BGR')
                # Current number of classes
                class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                class_fq = json.dumps(class_fq, indent = 4)
                class_fq = json.loads(class_fq)
                df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                    
                # Updating Inference results
                with st.container():
                    st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
                    st.markdown("<h3>Detected objects in curret Frame</h3>", unsafe_allow_html=True)
                    st.dataframe(df_fq, use_container_width=True)
        
    # Video
    # if options == 'Video':
    #     upload_video_file = st.sidebar.file_uploader(
    #         'Upload Video', type=['mp4', 'avi', 'mkv'])
    #     if upload_video_file is not None:
    #         pred = st.checkbox(f'Predict Using {model_type}')
    #         tfile = tempfile.NamedTemporaryFile(delete=False)
    #         tfile.write(upload_video_file.read())
    #         cap = cv2.VideoCapture(tfile.name)
    #         # if pred:



    def is_key_frame(prev_frame, curr_frame, threshold=300000):
        diff = cv2.absdiff(prev_frame, curr_frame)
        non_zero_count = np.count_nonzero(diff)
        return non_zero_count > threshold

    def is_key_frame(prev_frame, curr_frame, threshold=300000):
        diff = cv2.absdiff(prev_frame, curr_frame)
        non_zero_count = np.count_nonzero(diff)
        return non_zero_count > threshold

    # 原有的代码
    if options == 'Video':
        upload_video_file = st.sidebar.file_uploader(
            'Upload Video', type=['mp4', 'avi', 'mkv'])
        if upload_video_file is not None:
            pred = st.checkbox(f'Predict Using {model_type}')
            extract_key_frames = st.checkbox('Extract Key Frames')  # 新增的关键帧提取选项
            key_frames = []  # 存储关键帧的列表
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(upload_video_file.read())
            cap = cv2.VideoCapture(tfile.name)
        if (cap is not None) and pred:
            stframe1 = st.empty()
            stframe2 = st.empty()
            stframe3 = st.empty()
            prev_frame = None
            while True:
                success, img = cap.read()
                if not success:
                    st.error(
                        f"{options} NOT working\nCheck {options} properly!!",
                        icon="🚨"
                    )
                    break
                if extract_key_frames:
                    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None
                    if prev_frame is not None and gray_frame is not None and is_key_frame(prev_frame, gray_frame):
                        key_frames.append(img)
                    prev_frame = gray_frame
                img, current_no_class = get_yolo(img, model_type, model, confidence, color_pick_list, class_labels, draw_thick)
                FRAME_WINDOW.image(img, channels='BGR')

                # 检查 current_no_class 是否存在
                if current_no_class:
                    class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                    class_fq = json.dumps(class_fq, indent=4)
                    class_fq = json.loads(class_fq)
                    df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])

                    
                # 计算FPS
                c_time = time.time()
                fps = 1 / (c_time - p_time)
                p_time = c_time

                # 更新推理结果
                get_system_stat(stframe1, stframe2, stframe3, fps, df_fq)


            # if extract_key_frames:
            #     st.write(f'Extracted {len(key_frames)} key frames.')
            #     for i, frame in enumerate(key_frames):
            #         st.image(frame, caption=f'Key Frame {i+1}', channels='BGR')


         #Web-cam
    if options == 'Webcam':
        cam_options = st.sidebar.selectbox('Webcam Channel',
                                           ('Select Channel', '0', '1', '2', '3'))  
        if not cam_options == 'Select Channel':
            pred = st.checkbox(f'Predict Using {model_type}')
            cap = cv2.VideoCapture(int(cam_options))
            if not cap.isOpened():
                st.error("Error: Could not open webcam.")
            else:
                st.success(f"Webcam channel {cam_options} opened successfully.")
    #
    #         if not cam_options == 'Select Channel':
    #             pred = st.checkbox(f'Predict Using {model_type}')
    #             cap = cv2.VideoCapture(int(cam_options))
        # RTSP
        # if options == 'RTSP':
        #     rtsp_url = st.sidebar.text_input(
        #         'RTSP URL:',
        #         'eg: rtsp://admin:name6666@198.162.1.58/cam/realmonitor?channel=0&subtype=0'
        #     )
        #     pred = st.checkbox(f'Predict Using {model_type}')
        #     cap = cv2.VideoCapture(rtsp_url)
if (cap != None) and pred:
    stframe1 = st.empty()
    stframe2 = st.empty()
    stframe3 = st.empty()
    while True:
        success, img = cap.read()
        if not success:
            st.error(
                f"{options} NOT working\nCheck {options} properly!!",
                icon="🚨"
            )
            break
        img, current_no_class = get_yolo(img, model_type, model, confidence, color_pick_list, class_labels, draw_thick)
        FRAME_WINDOW.image(img, channels='BGR')
        # FPS
        # c_time = time.time()
        # fps = 1 / (c_time - p_time)
        # p_time = c_time
        
        # # 检查 current_no_class 是否存在
        # if current_no_class:
        #     class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
        #     class_fq = json.dumps(class_fq, indent=4)
        #     class_fq = json.loads(class_fq)
        #     df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
        
        # # Updating Inference results
        # get_system_stat(stframe1, stframe2, stframe3, fps, df_fq)
