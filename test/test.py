import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 모델 로드 (YOLOv8 사전 학습된 모델 사용)
model = YOLO('yolov8n.pt')  # 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt' 중 선택

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

# 기준 물체의 크기를 기반으로 다른 물체 크기 계산
for i, (x1, y1, x2, y2) in enumerate(boxes):
    width = x2 - x1
    height = y2 - y1

    cm_width = width / base_width * base_cm_width
    cm_height = height / base_height * base_cm_height
    cm_depth = base_cm_depth * (cm_width / base_cm_width)

    print(f"[{i}] : 가로: {cm_width:.2f}cm, 세로: {cm_height:.2f}cm, 높이(깊이): {cm_depth:.2f}cm")

    # Canny Edge Detection 적용
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 모서리 점 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 모서리 찾기 (가장 많은 점을 가진 모서리)
    largest_contour = max(contours, key=len)

    # 모서리 점 출력
    print(f"[{i}] 모서리 점:")
    for j, point in enumerate(largest_contour):
        print(f"  - ({point[0][0]}, {point[0][1]})")

        # 모서리 점 이미지에 표시
        cv2.circle(result_image, (x1 + point[0][0], y1 + point[0][1]), 3, (0, 0, 255), -1)

        # 모서리 길이 계산 및 표시
        if j > 0:
            previous_point = largest_contour[j - 1][0]
            edge_length = ((point[0][0] - previous_point[0])**2 + (point[0][1] - previous_point[1])**2)**0.5
            cv2.putText(result_image, f"{edge_length:.2f}", (x1 + (point[0][0] + previous_point[0]) // 2,
                                        y1 + (point[0][1] + previous_point[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

# 결과 이미지 표시
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # 축 눈금 숨기기
plt.show()