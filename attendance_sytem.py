import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from PIL import Image
import numpy as np
import time

# --- CẤU HÌNH ---
DB_PATH = 'face_database.pt'
THRESHOLD = 0.75  # Ngưỡng nhận diện (càng thấp càng khắt khe)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_database():
    try:
        # Load dictionary {name: vector}
        database = torch.load(DB_PATH, map_location=device)
        
        # Chuyển đổi sang dạng List để dễ xử lý bằng Tensor
        names = list(database.keys())
        # Stack các vector lại thành 1 ma trận [N_people, 512] để tính toán cho nhanh
        embeddings = torch.stack(list(database.values())).to(device)
        
        print(f"Đã load dữ liệu của {len(names)} người: {names}")
        return names, embeddings
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{DB_PATH}'. Hãy chạy create_db_multi.py trước!")
        return None, None

def main():
    # 1. Load Models & Data
    print("Loading models...")
    # MTCNN chạy CPU cho ổn định webcam
    mtcnn = MTCNN(keep_all=True, device='cpu', min_face_size=30, thresholds=[0.6, 0.7, 0.7])
    
    # Resnet chạy GPU (nếu có)
    resnet = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    resnet.eval()

    names_list, embeddings_matrix = load_database()
    if names_list is None: return

    # 2. Mở Camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Hệ thống điểm danh đã sẵn sàng. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Detect faces
        boxes, _ = mtcnn.detect(img_pil)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                # Padding
                x1, y1 = max(0, x1-10), max(0, y1-10)
                x2, y2 = min(frame.shape[1], x2+10), min(frame.shape[0], y2+10)

                # Crop & Process
                if x1 >= x2 or y1 >= y2: continue
                face_img = img_pil.crop((x1, y1, x2, y2)).resize((160, 160))
                
                face_tensor = fixed_image_standardization(
                    torch.tensor(np.array(face_img)).permute(2, 0, 1).float()
                ).unsqueeze(0).to(device)

                # --- LOGIC NHẬN DIỆN (1 vs N) ---
                with torch.no_grad():
                    current_emb = resnet(face_tensor) # [1, 512]
                
                # Tính khoảng cách Euclidean từ vector hiện tại đến TOÀN BỘ database
                # (Ma trận [N, 512] trừ Vector [1, 512] -> Broadcasting)
                dist_matrix = (embeddings_matrix - current_emb).norm(p=2, dim=1)
                
                # Tìm khoảng cách nhỏ nhất
                min_dist, min_idx = torch.min(dist_matrix, 0)
                min_dist = min_dist.item()
                idx = min_idx.item()

                # Kiểm tra ngưỡng
                if min_dist < THRESHOLD:
                    name = names_list[idx]
                    color = (0, 255, 0) # Xanh lá
                    label = f"{name} ({min_dist:.2f})"
                    
                    # TODO: Tại đây bạn có thể thêm code ghi vào Excel/Database để điểm danh
                    # if name not in checked_list: save_to_excel(name)
                else:
                    name = "Unknown"
                    color = (0, 0, 255) # Đỏ
                    label = f"Unknown ({min_dist:.2f})"

                # Vẽ
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Class Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()