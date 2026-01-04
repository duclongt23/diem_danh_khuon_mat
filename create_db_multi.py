import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import glob

# --- CẤU HÌNH ---
DATA_DIR = 'dataset_train' # Folder chứa các folder con của từng người
DB_PATH = 'face_database.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_multi_user_db():
    print(f"Đang chạy trên: {device}")
    
    # 1. Load Models
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    resnet.eval()

    # Dictionary để lưu database: { "Tên_Người": Vector_Trung_Bình }
    database = {}

    # 2. Duyệt qua từng folder con trong dataset_train
    if not os.path.exists(DATA_DIR):
        print(f"Lỗi: Không tìm thấy folder '{DATA_DIR}'")
        return

    # Lấy danh sách các folder con (tên người)
    subfolders = [f.path for f in os.scandir(DATA_DIR) if f.is_dir()]

    print(f"Tìm thấy {len(subfolders)} người cần xử lý.")

    for folder in subfolders:
        person_name = os.path.basename(folder) # Lấy tên folder làm tên người
        print(f"\n--- Đang xử lý: {person_name} ---")
        
        image_paths = glob.glob(os.path.join(folder, "*.*"))
        embeddings = []

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Cắt mặt & Chuẩn hóa
                face_tensor = mtcnn(img)
                
                if face_tensor is not None:
                    face_tensor = face_tensor.to(device).unsqueeze(0)
                    
                    # Tính vector
                    with torch.no_grad():
                        emb = resnet(face_tensor).cpu() # Đưa về CPU để lưu trữ cho nhẹ
                    embeddings.append(emb)
            except Exception as e:
                pass # Bỏ qua ảnh lỗi

        if len(embeddings) > 0:
            # Tính vector trung bình của người này
            embeddings = torch.cat(embeddings)
            mean_embedding = embeddings.mean(dim=0)
            
            # Lưu vào dictionary
            database[person_name] = mean_embedding
            print(f"-> Đã thêm {person_name} (từ {len(embeddings)} ảnh).")
        else:
            print(f"-> Cảnh báo: Không tìm thấy mặt nào trong folder {person_name}!")

    # 3. Lưu database xuống file
    if len(database) > 0:
        torch.save(database, DB_PATH)
        print("\n" + "="*40)
        print(f"HOÀN TẤT! Đã lưu dữ liệu của {len(database)} người vào '{DB_PATH}'")
        print("="*40)
    else:
        print("Không tạo được database nào.")

if __name__ == '__main__':
    create_multi_user_db()