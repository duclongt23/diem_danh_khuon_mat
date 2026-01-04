import cv2
import os

def extract_images_from_video(video_path, output_folder, interval=10):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Cứ mỗi 'interval' frame thì lưu 1 lần
        if count % interval == 0:
            cv2.imwrite(f"{output_folder}/img_{saved_count}.jpg", frame)
            saved_count += 1
        count += 1
    
    cap.release()
    print(f"Đã thu thập {saved_count} ảnh từ video.")

extract_images_from_video('duc_long.mp4', 'dataset_train/0_me', interval=30)