import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import glob
import sys

# --- CẤU HÌNH ---
DATA_DIR = 'dataset_train'   # Folder gốc chứa ảnh
DB_PATH = 'face_database.pt' # File database hiện tại
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def add_single_user():
    print(f"--- TOOL THÊM NGƯỜI MỚI VÀO DATABASE ({device}) ---")
    
    # 1. Nhập tên người mới (Tương ứng tên folder)
    new_name = input("Nhập tên folder của người mới (VD: NguyenVanZ): ").strip()
    
    user_folder = os.path.join(DATA_DIR, new_name)
    if not os.path.exists(user_folder):
        print(f"[LỖI]: Không tìm thấy folder '{user_folder}'. Hãy tạo folder và copy ảnh vào trước.")
        return

    # 2. Load Database cũ
    if os.path.exists(DB_PATH):
        print("Đang load database cũ...")
        database = torch.load(DB_PATH, map_location=device)
        if new_name in database:
            ans = input(f"Cảnh báo: '{new_name}' đã có trong DB. Bạn có muốn ghi đè không? (y/n): ")
            if ans.lower() != 'y':
                print("Hủy thao tác.")
                return
    else:
        print("Chưa có database, sẽ tạo mới.")
        database = {}

    # 3. Khởi tạo Model (Chỉ cần làm 1 lần)
    print("Đang khởi tạo model...")
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    resnet.eval()

    # 4. Xử lý ảnh của người mới
    image_paths = glob.glob(os.path.join(user_folder, "*.*"))
    print(f"Tìm thấy {len(image_paths)} ảnh. Đang tính toán vector...")
    
    embeddings = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            # Cắt mặt
            face_tensor = mtcnn(img)
            
            if face_tensor is not None:
                face_tensor = face_tensor.to(device).unsqueeze(0)
                # Tính embedding
                with torch.no_grad():
                    emb = resnet(face_tensor).cpu()
                embeddings.append(emb)
                sys.stdout.write(".") # Hiển thị tiến trình
                sys.stdout.flush()
        except Exception:
            pass

    print("\n")
    
    # 5. Lưu vào Database
    if len(embeddings) > 0:
        embeddings = torch.cat(embeddings)
        mean_embedding = embeddings.mean(dim=0)
        
        # Cập nhật Dictionary
        database[new_name] = mean_embedding
        
        # Lưu file
        torch.save(database, DB_PATH)
        print(f"[THÀNH CÔNG]: Đã thêm '{new_name}' vào database.")
        print(f"Tổng số người trong DB hiện tại: {len(database)}")
    else:
        print("[THẤT BẠI]: Không tìm thấy khuôn mặt nào trong ảnh cung cấp.")

if __name__ == '__main__':
    add_single_user()