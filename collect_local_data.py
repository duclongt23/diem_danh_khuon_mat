# file: collect_local_data.py
import cv2
import os
import time

# Tạo folder
DATA_DIR = 'dataset_train'
for lbl in ['person1']:
    os.makedirs(os.path.join(DATA_DIR, lbl), exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

print("--- THU THẬP DỮ LIỆU LOCAL ---")
print("Nhấn 'r' để lưu ảnh mặt")
print("Nhấn 'q' để thoát")

counts = {'count': 0}

while True:
    ret, frame = cap.read()
    if not ret: break
    
    display_frame = frame.copy()
    
    # Hiển thị số lượng
    cv2.putText(display_frame, f"Count: {counts['count']}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('Collect Data', display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Lưu ảnh Real
        filename = os.path.join(DATA_DIR, 'person1', f"person1_{int(time.time()*100)}.jpg")
        cv2.imwrite(filename, frame)
        counts['real'] += 1
        print(f"Saved Real: {filename}")

cap.release()
cv2.destroyAllWindows()