import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import os

# --- CẤU HÌNH ---
DB_PATH = 'model/face_database.pt' # File database chứa nhiều người
THRESHOLD = 0.75             # Ngưỡng nhận diện (tinh chỉnh từ 0.6 - 0.8)

# Thiết lập thiết bị
# MTCNN nên chạy CPU để xử lý ảnh tĩnh kích thước lớn ổn định hơn
DEVICE_MTCNN = torch.device('cpu') 
DEVICE_RESNET = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Biến toàn cục lưu model & data
mtcnn = None
resnet = None
db_names = []       # Danh sách tên người [Name1, Name2, ...]
db_embeddings = None # Ma trận vector [N_people, 512]

def initialize_system():
    global mtcnn, resnet, db_names, db_embeddings
    print(f"--- Đang khởi tạo hệ thống ---")
    
    # 1. Load Database
    if not os.path.exists(DB_PATH):
        print(f"[LỖI]: Không tìm thấy file '{DB_PATH}'")
        exit()
        
    print(f"1. Loading Database từ {DB_PATH}...")
    database = torch.load(DB_PATH, map_location=DEVICE_RESNET)
    
    # Chuyển đổi Dictionary thành Matrix để tính toán nhanh
    db_names = list(database.keys())
    # Stack gom các vector lại thành 1 Tensor duy nhất [N, 512]
    db_embeddings = torch.stack(list(database.values())).to(DEVICE_RESNET)
    
    print(f"   -> Đã load {len(db_names)} người: {db_names}")

    # 2. Load Models
    print("2. Loading MTCNN & FaceNet...")
    mtcnn = MTCNN(keep_all=True, device=DEVICE_MTCNN, min_face_size=30)
    
    resnet = InceptionResnetV1(pretrained='vggface2', classify=False).to(DEVICE_RESNET)
    resnet.eval()
    
    print("--- Hệ thống sẵn sàng ---\n")

def recognize_faces_in_image(image_path, output_path=None):
    if mtcnn is None: initialize_system()
    
    # 1. Đọc ảnh
    if not os.path.exists(image_path):
        print(f"Không tìm thấy ảnh: {image_path}")
        return

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("File ảnh lỗi.")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    display_img = img_bgr.copy()

    # 2. Phát hiện khuôn mặt
    print(f"Đang xử lý ảnh: {image_path}...")
    boxes, _ = mtcnn.detect(img_pil)

    if boxes is None:
        print("-> Không tìm thấy khuôn mặt nào.")
    else:
        print(f"-> Tìm thấy {len(boxes)} khuôn mặt.")
        
        # 3. Duyệt qua từng mặt
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Padding an toàn
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img_pil.width, x2 + padding)
            y2 = min(img_pil.height, y2 + padding)

            # Crop & Resize
            face_img = img_pil.crop((x1, y1, x2, y2))
            face_img = face_img.resize((160, 160))
            
            # Convert to Tensor
            face_tensor = fixed_image_standardization(
                torch.tensor(np.array(face_img)).permute(2, 0, 1).float()
            ).unsqueeze(0).to(DEVICE_RESNET)

            # --- MATCHING LOGIC (1 vs N) ---
            with torch.no_grad():
                current_emb = resnet(face_tensor) # [1, 512]
            
            # Tính khoảng cách Euclidean với TOÀN BỘ Database cùng lúc
            # (Broadcasting: [N, 512] - [1, 512])
            dist_matrix = (db_embeddings - current_emb).norm(p=2, dim=1)
            
            # Tìm người giống nhất (khoảng cách nhỏ nhất)
            min_dist, idx = torch.min(dist_matrix, 0)
            min_dist = min_dist.item()
            idx = idx.item()

            # Quyết định danh tính
            if min_dist < THRESHOLD:
                name = db_names[idx]
                color = (0, 255, 0) # Green
                text = f"{name} ({min_dist:.2f})"
            else:
                name = "Unknown"
                color = (0, 0, 255) # Red
                text = f"Unknown ({min_dist:.2f})"

            print(f" - Mặt {i+1}: {text}")

            # Vẽ lên ảnh
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_img, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 4. Hiển thị kết quả
    # Resize nếu ảnh quá to để vừa màn hình laptop
    h, w = display_img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        display_img = cv2.resize(display_img, (0,0), fx=scale, fy=scale)

    cv2.imshow(f"Result - {os.path.basename(image_path)}", display_img)
    
    if output_path:
        cv2.imwrite(output_path, display_img)
        print(f"Đã lưu kết quả tại: {output_path}")
    
    print("Nhấn phím bất kỳ để đóng cửa sổ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Thay đường dẫn ảnh bạn muốn test vào đây
    TEST_IMAGE = "family.jpg" 
    
    # Tạo file giả nếu chưa có
    if not os.path.exists(TEST_IMAGE):
        print(f"File {TEST_IMAGE} không tồn tại. Vui lòng copy ảnh vào để test.")
    else:
        recognize_faces_in_image(TEST_IMAGE, output_path="result_multi.jpg")