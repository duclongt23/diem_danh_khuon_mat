import cv2
import torch
import numpy as np
from PIL import Image
from collections import deque
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization

# --- Import từ Repo Silent-Face-Anti-Spoofing ---
import sys
sys.path.append('./src') 
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage

# --- CẤU HÌNH ---
DB_PATH = 'model/face_database.pt'
SPOOF_THRESHOLD = 0.5       
RECOGNITION_THRESHOLD = 0.70
VOTING_WINDOW = 20            
REAL_RATIO = 0.6              

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo Anti-Spoof
model_test = AntiSpoofPredict(0)
image_cropper = CropImage()
model_dir = "./resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"

# Bộ nhớ đệm lưu kết quả liveness
liveness_history = deque(maxlen=VOTING_WINDOW)

def check_liveness(frame, face_box):
    """
    Hàm này trả về:
    - is_real: True/False dựa trên ngưỡng threshold
    - real_score: Độ tin tưởng là người thật (0.0 -> 1.0)
    """
    x, y, w, h = face_box
    try:
        param = {
            "org_img": frame, "bbox": [x, y, w, h],
            "scale": 2.7, "out_w": 80, "out_h": 80, "crop": True,
        }
        img = image_cropper.crop(**param)
        prediction = model_test.predict(img, model_dir)
        
        # prediction[0][1] là xác suất lớp Real. 
        # Chia 2 để đưa về thang điểm 0-1 (do output gốc của model này là thang 0-2)
        real_score = prediction[0][1] 
        
        is_real = True if real_score > SPOOF_THRESHOLD else False
        return is_real, real_score
    except:
        return False, 0.0

def load_database():
    try:
        database = torch.load(DB_PATH, map_location=device)
        return list(database.keys()), torch.stack(list(database.values())).to(device)
    except:
        print(f"Lỗi: Không tìm thấy file {DB_PATH}")
        return None, None

def main():
    # Tăng min_face_size lên 80 để buộc người dùng không đứng quá sát cam
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=80) 
    resnet = InceptionResnetV1(pretrained='vggface2', classify=False).to(device).eval()
    names_list, embeddings_matrix = load_database()
    if names_list is None: return
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(Image.fromarray(img_rgb))

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                w_face, h_face = x2 - x1, y2 - y1

                # 1. Lấy Real Score hiện tại
                is_real_frame, current_real_score = check_liveness(frame, [x1, y1, w_face, h_face])
                liveness_history.append(is_real_frame)

                # 2. Tính tỉ lệ ổn định (Voting)
                ratio = liveness_history.count(True) / len(liveness_history)

                if ratio >= REAL_RATIO:
                    # --- XỬ LÝ NHẬN DIỆN ---
                    face_img = Image.fromarray(img_rgb).crop((x1, y1, x2, y2)).resize((160, 160))
                    face_tensor = fixed_image_standardization(
                        torch.tensor(np.array(face_img)).permute(2, 0, 1).float()
                    ).unsqueeze(0).to(device)

                    with torch.no_grad():
                        emb = resnet(face_tensor)
                    
                    dist = (embeddings_matrix - emb).norm(p=2, dim=1)
                    min_dist, min_idx = torch.min(dist, 0)
                    dist_val = min_dist.item()

                    if dist_val < RECOGNITION_THRESHOLD:
                        label = f"{names_list[min_idx.item()]} (Dist: {dist_val:.2f})"
                        color = (0, 255, 0)
                    else:
                        label = f"Unknown (Dist: {dist_val:.2f})"
                        color = (0, 255, 255)
                else:
                    # --- BÁO GIẢ MẠO ---
                    label = "SPOOF DETECTED"
                    color = (0, 0, 255)

                # --- HIỂN THỊ DEBUG ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Dòng 1: Trạng thái và Khoảng cách nhận diện
                cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Dòng 2: Chỉ số chống giả mạo quan trọng
                debug_txt = f"RealScore: {current_real_score:.2f} | Ratio: {ratio:.2f}"
                cv2.putText(frame, debug_txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Liveness System Debug Mode', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()