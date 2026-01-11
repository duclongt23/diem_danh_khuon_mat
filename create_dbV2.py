import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import glob
import sys

# --- CẤU HÌNH ---
DATA_DIR = 'dataset_train' # Folder gốc chứa các folder con (tên người)
DB_PATH = 'model/multi_face_database.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_multi_user_db_knn():
    print(f"--- TẠO DATABASE CHO k-NN (Running on {device}) ---")
    
    # 1. Kiểm tra folder dữ liệu
    if not os.path.exists(DATA_DIR):
        print(f"[LỖI]: Không tìm thấy folder '{DATA_DIR}'.")
        print("Hãy tạo folder này và bỏ ảnh vào theo cấu trúc: dataset_train/Ten_Nguoi/anh1.jpg")
        return

    # 2. Khởi tạo Models
    print("Đang load models...")
    # keep_all=False: Vì ảnh train thường chỉ có 1 mặt người
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    
    # InceptionResnetV1 lấy vector 512 chiều
    resnet = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    resnet.eval()

    database = {} # Dictionary: { "Tên": Tensor[N_images, 512] }

    # 3. Duyệt qua từng người
    subfolders = [f.path for f in os.scandir(DATA_DIR) if f.is_dir()]
    print(f"Tìm thấy {len(subfolders)} người trong dataset.")

    for folder in subfolders:
        person_name = os.path.basename(folder)
        print(f"\n> Đang xử lý: {person_name}")
        
        image_paths = glob.glob(os.path.join(folder, "*.*"))
        
        person_embeddings = [] # List tạm để chứa vector của người này
        
        for img_path in image_paths:
            try:
                # Mở ảnh
                img = Image.open(img_path).convert('RGB')
                
                # Cắt mặt & Chuẩn hóa (MTCNN trả về Tensor [3, 160, 160])
                face_tensor = mtcnn(img)
                
                if face_tensor is not None:
                    # Thêm batch dimension -> [1, 3, 160, 160]
                    face_tensor = face_tensor.to(device).unsqueeze(0)
                    
                    # Tính Embedding -> [1, 512]
                    with torch.no_grad():
                        emb = resnet(face_tensor).cpu() # Đưa về CPU để lưu RAM
                    
                    person_embeddings.append(emb)
                    sys.stdout.write(".") # Hiển thị tiến trình
                    sys.stdout.flush()
                else:
                    # Không tìm thấy mặt trong ảnh train
                    pass 
            except Exception as e:
                print(f"\nLỗi ảnh {os.path.basename(img_path)}: {e}")

        # 4. Lưu vào Database
        if len(person_embeddings) > 0:
            # Gộp list lại thành 1 Tensor duy nhất [Số_ảnh, 512]
            # QUAN TRỌNG: Không dùng .mean() ở đây nữa!
            final_tensor = torch.cat(person_embeddings)
            
            database[person_name] = final_tensor
            print(f"\n   -> Xong! Lưu {final_tensor.shape[0]} vector (Shape: {final_tensor.shape})")
        else:
            print(f"\n   -> Cảnh báo: Không tìm thấy khuôn mặt nào hợp lệ cho {person_name}")

    # 5. Lưu file
    if len(database) > 0:
        torch.save(database, DB_PATH)
        print("\n" + "="*40)
        print(f"HOÀN TẤT! Database đã lưu tại '{DB_PATH}'")
        print(f"Sẵn sàng cho thuật toán k-NN.")
        print("="*40)
    else:
        print("\n[THẤT BẠI] Không tạo được dữ liệu nào.")

if __name__ == '__main__':
    create_multi_user_db_knn()