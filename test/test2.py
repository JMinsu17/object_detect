import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

# MiDaS 모델 로드
try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
except Exception as e:
    print(f"Error loading MiDaS model: {e}")
    exit(1)

midas.to("cuda" if torch.cuda.is_available() else "cpu")
midas.eval()

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

# 이미지 경로 설정
image_path = '/Users/tobesmart/Documents/object_detect/images/3.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to read image from '{image_path}'")
    exit(1)

# YOLOv8 모델 적용
results = model.predict(source=image, save=False, conf=0.25)

# 모델에서 인식 가능한 클래스 이름 확인
class_names = model.names

# 결과 이미지 복사본 생성
result_image = image.copy()

# 모든 클래스에 대한 Bounding Box 그리기 및 크기 출력
boxes = []

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = class_names[class_id]
        confidence = box.conf[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1
        boxes.append([x1, y1, x2, y2])

        # Bounding Box 및 텍스트 표시
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_image, f"{label}: {confidence:.2f} (W: {width}, H: {height})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# 크기의 기준이 되는 물체의 번호를 입력받기
size_base_index = int(input("크기의 기준이 되는 물체의 번호를 입력하세요: "))
if size_base_index < 0 or size_base_index >= len(boxes):
    raise ValueError("잘못된 번호입니다.")

# 기준 물체의 크기 계산
base_x1, base_y1, base_x2, base_y2 = boxes[size_base_index]
base_width = base_x2 - base_x1
base_height = base_y2 - base_y1

print(f"선택한 객체의 크기\n가로: {base_width} 픽셀, 세로: {base_height} 픽셀")
base_cm_width = float(input("선택한 물체의 가로 길이를 cm 단위로 입력해주세요: "))
base_cm_height = float(input("선택한 물체의 세로 길이를 cm 단위로 입력해주세요: "))
base_cm_depth = float(input("선택한 물체의 높이(깊이)를 cm 단위로 입력해주세요: "))

# MiDaS를 사용한 깊이 추정
transform = T.Compose([
    T.ToTensor(),
    T.Resize(384, interpolation=T.InterpolationMode.BICUBIC),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_batch = transform(image).to("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    # 추론 및 후처리
    prediction = midas(input_batch.unsqueeze(0))
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()

# 깊이 맵 정규화
output_display = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
output_display = cv2.cvtColor(output_display, cv2.COLOR_GRAY2BGR)

# 기준 물체의 크기를 기반으로 다른 물체 크기 및 깊이 계산
for i, (x1, y1, x2, y2) in enumerate(boxes):
    width = x2 - x1
    height = y2 - y1

    cm_width = width / base_width * base_cm_width
    cm_height = height / base_height * base_cm_height
    cm_depth = base_cm_depth * (cm_width / base_cm_width)

    # 객체의 평균 깊이 계산
    avg_depth = depth_map[y1:y2, x1:x2].mean()

    print(f"[{i}] : 가로: {cm_width:.2f}cm, 세로: {cm_height:.2f}cm, 높이(깊이): {cm_depth:.2f}cm, 평균 깊이: {avg_depth:.2f}")

# 결과 이미지 표시
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title('Object Detection')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_display)
plt.title('Depth Estimation')
plt.axis('off')

plt.show()