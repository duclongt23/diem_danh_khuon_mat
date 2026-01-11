import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization

# --- Import từ Repo Silent-Face-Anti-Spoofing ---
import sys
sys.path.append('./src') 
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# --- CẤU HÌNH ---
DB_PATH = 'model/face_database.pt' # Đường dẫn file database
SPOOF_THRESHOLD = 0.5            
RECOGNITION_THRESHOLD = 0.70      
DEVICE_ID = 0                      

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def enhance_face(img_patch):
    # Chuyển sang LAB để cân bằng độ sáng mà không đổi màu sắc
    lab = cv2.cvtColor(img_patch, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# --- KHỞI TẠO ANTI-SPOOFING ---
# Load model Anti-Spoofing (MiniFASNetV2)
model_test = AntiSpoofPredict(DEVICE_ID)
image_cropper = CropImage()

def check_liveness(frame, face_box):
    """
    Trả về: (is_real: bool, score: float, label_text: str)
    """
    # Lấy thông số box
    x, y, w, h = face_box
    # Tính toán crop rộng (scale 2.7) theo yêu cầu của model Anti-Spoof
    # Model này cần nhìn thấy cả viền xung quanh mặt để phát hiện điện thoại/ipad
    prediction = np.zeros((1, 3))
    
    # Cần scale box rộng ra 2.7 lần
    model_dir = "./resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
    
    # Hàm crop của thư viện yêu cầu format: [x, y, w, h]
    # frame gốc (numpy array)
    try:
        image_bbox = [x, y, w, h] 
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": 2.7, # Quan trọng: Model train với scale 2.7
            "out_w": 80,
            "out_h": 80,
            "crop": True,
        }
        img = image_cropper.crop(**param)
        
        # Dự đoán
        prediction += model_test.predict(img, model_dir)
        
        # Kết quả: Class 0 = Fake, Class 1 = Real
        label = np.argmax(prediction)
        value = prediction[0][label] 
        
        # Logic đánh giá
        if label == 1 and value > SPOOF_THRESHOLD:
            return True, value, "REAL"
        else:
            return False, value, "FAKE"
            
    except Exception as e:
        # Nếu crop bị lỗi (ra ngoài khung hình)
        return False, 0.0, "Error"

def load_database():
    try:
        database = torch.load(DB_PATH, map_location=device)
        names = list(database.keys())
        embeddings = torch.stack(list(database.values())).to(device)
        print(f"-> Đã load {len(names)} người dùng.")
        return names, embeddings
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy {DB_PATH}")
        return None, None

def main():
    # 1. Load Recognition Models
    print("Loading models...")
    # MTCNN để detect mặt (CPU cho ổn định, hoặc GPU nếu mạnh)
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=30, thresholds=[0.6, 0.7, 0.7])
    
    # InceptionResnetV1 để nhận diện
    resnet = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    resnet.eval()

    names_list, embeddings_matrix = load_database()
    if names_list is None: return

    # 2. Camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Dùng DSHOW trên Windows cho nhanh
    cap.set(3, 640)
    cap.set(4, 480)

    print("\n--- STARTING SYSTEM ---")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # MTCNN cần ảnh RGB (PIL hoặc Numpy RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Detect Faces
        boxes, _ = mtcnn.detect(img_pil)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                w_box = x2 - x1
                h_box = y2 - y1

                # --- BƯỚC 1: ANTI-SPOOFING (Liveness Check) ---
                # Truyền frame gốc BGR (opencv) và box vào
                
                is_real, liveness_score, liveness_label = check_liveness(frame, [x1, y1, w_box, h_box])

                if not is_real:
                    # Nếu là GIẢ -> Vẽ đỏ, không nhận diện tên
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"SPOOF: {liveness_score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    continue # Bỏ qua bước nhận diện tên

                # --- BƯỚC 2: FACE RECOGNITION (Chỉ chạy khi là mặt thật) ---
                # Crop mặt cho FaceNet (Padding nhẹ)
                p = 10 
                x1_p, y1_p = max(0, x1-p), max(0, y1-p)
                x2_p, y2_p = min(frame.shape[1], x2+p), min(frame.shape[0], y2+p)
                
                if x1_p >= x2_p or y1_p >= y2_p: continue

                face_img = img_pil.crop((x1_p, y1_p, x2_p, y2_p)).resize((160, 160))
                
                face_tensor = fixed_image_standardization(
                    torch.tensor(np.array(face_img)).permute(2, 0, 1).float()
                ).unsqueeze(0).to(device)

                with torch.no_grad():
                    current_emb = resnet(face_tensor)
                
                # Tính khoảng cách
                dist_matrix = (embeddings_matrix - current_emb).norm(p=2, dim=1)
                min_dist, min_idx = torch.min(dist_matrix, 0)
                min_dist = min_dist.item()

                if min_dist < RECOGNITION_THRESHOLD:
                    name = names_list[min_idx.item()]
                    color = (0, 255, 0) # Xanh lá (Real + Recognized)
                    status = f"{name} ({min_dist:.2f})"
                else:
                    name = "Unknown"
                    color = (0, 255, 255) # Vàng (Real + Unknown)
                    status = f"Unknown ({min_dist:.2f})"

                # Vẽ kết quả
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"REAL: {liveness_score:.2f}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, status, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Secure Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()