from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
import uvicorn
import io

app = FastAPI()
model = YOLO('yolov8n.pt')  # Thay 'yolov8n.pt' bằng đường dẫn model của bạn

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        # Đọc ảnh từ file upload
        image_data = await file.read()  # Đọc dữ liệu ảnh từ request
        image = Image.open(io.BytesIO(image_data))  # Chuyển đổi dữ liệu byte thành ảnh PIL

        # Phát hiện đối tượng trong ảnh
        results = model(image)

        # Lấy các bounding box, độ tin cậy, và lớp từ kết quả
        detections = []

        # Lấy các bounding box, độ tin cậy và lớp từ kết quả YOLOv8
        boxes = results[0].boxes
        for i in range(len(boxes.xyxy)):  # Duyệt qua từng bounding box
            box = boxes.xyxy[i].cpu().numpy()  # Chuyển tọa độ bounding box thành numpy array
            x1, y1, x2, y2 = box.astype(int)  # Chuyển sang kiểu int

            # Lấy độ tin cậy của prediction (confidence)
            conf = boxes.conf[i].item()  # Chuyển giá trị thành kiểu float

            # Lấy lớp của đối tượng (class)
            cls = boxes.cls[i].item()  # Chuyển giá trị thành kiểu int

            # Lưu thông tin phát hiện vào danh sách
            detection = {
                "xmin": int(x1),
                "ymin": int(y1),
                "xmax": int(x2),
                "ymax": int(y2),
                "confidence": float(conf),  # Chuyển giá trị thành float nếu cần
                "class": int(cls)           # Chuyển giá trị thành int
            }
            detections.append(detection)

        if not detections:
            return JSONResponse(status_code=200, content={"message": "No objects detected."})

        # Trả về kết quả phát hiện
        return {"detections": detections}

    except Exception as e:
        # Trả về lỗi nếu có
        return JSONResponse(status_code=500, content={"error": str(e)})

# Endpoint kiểm tra API
@app.get("/")
def home():
    return {"message": "YOLOv8 API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
