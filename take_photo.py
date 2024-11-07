import cv2
import os

def generate_dataset(your_name):
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") ## Sửa đường dẫn để chạy file 
    
    def face_cropped(img, padding=20):  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        for (x, y, w, h) in faces:
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)
            cropped_face = img[y1:y2, x1:x2]
        return cropped_face


    cap = cv2.VideoCapture(0)
    img_id = 0 ## Sửa để thêm ảnh

    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id += 1
            face = cv2.resize(face_cropped(frame), (200, 200))  # Lưu ảnh màu
            file_name_path = f"{your_name}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Cropped Face", face)
        
        # Nhấn Enter hoặc đạt số lượng ảnh cần thiết thì thoát
        if cv2.waitKey(1) == 13 or img_id == 200:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed...")

generate_dataset("") ##Điền tên
