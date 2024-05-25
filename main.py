import random
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

min_area = 1000
conf_threshold = 0.1  # 신뢰도 임계값
iou_threshold = 0.7  # IoU 임계값 낮을수록 큰걸 인식 높을수록 자잘한것도 같이 인식
model_path = "model/yolov8l-oiv7.pt"

# YOLO 모델 로드 함수
def load_yolo_model():
    model = YOLO(model_path)
    return model

# MiDaS 모델 로드 함수
def load_midas_model():
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")  # DPT_Hybrid 모델 사용
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    return model, transform

# 객체 인식 함수
def detect_objects(img, model, conf_threshold, iou_threshold):
    results = model.predict(img, conf=conf_threshold, iou=iou_threshold)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    return boxes, classes, results

# 깊이 추정 함수
def estimate_depth(img, model, transform, device):
    input_batch = transform(img).to(device)  # GPU 사용
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()

    # 깊이 맵 정규화
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    return depth_map

# 깊이 맵에서 객체 3D 크기 계산 함수
def calculate_object_dimensions(box, depth_map):
    x1, y1, x2, y2 = map(int, box[:4])
    object_depth = depth_map[y1:y2, x1:x2]
    median_depth = np.median(object_depth)

    width = x2 - x1
    height = y2 - y1
    depth = median_depth * 100  # 정규화된 깊이를 실제 깊이 값으로 변환 (예: 100을 곱함)

    return width, height, depth, x1, y1, x2, y2

# 큰 물체 필터링 함수
def filter_large_objects(boxes, classes, min_area):
    filtered_boxes = []
    filtered_classes = []
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box[:4])
        area = (x2 - x1) * (y2 - y1)
        if area >= min_area:
            filtered_boxes.append(box)
            filtered_classes.append(cls)
    return np.array(filtered_boxes), np.array(filtered_classes)

# 메인 분석 함수
def analyze_image(image_data):
    # 모델 로드
    yolo_model = load_yolo_model()
    midas_model, midas_transform = load_midas_model()

    # GPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas_model.to(device)

    # 이미지 로드
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("이미지를 로드할 수 없습니다.")
        return None

    # 객체 인식
    boxes, classes, results = detect_objects(img, yolo_model, conf_threshold, iou_threshold)

    # 큰 물체 필터링
    boxes, classes = filter_large_objects(boxes, classes, min_area)

    # 깊이 추정
    depth_map = estimate_depth(img, midas_model, midas_transform, device)

    # 각 객체의 3D 크기 계산 및 출력
    for i, (box, class_id) in enumerate(zip(boxes, classes)):
        width, height, depth, x1, y1, x2, y2 = calculate_object_dimensions(box, depth_map)
        class_name = results[0].names[int(class_id)]
        label = f"{class_name} W: {width}, H: {height}, D: {depth:.2f}"

        # 콘솔에 로그 출력
        print(f"Object: {class_name}, Width: {width}, Height: {height}, Depth: {depth:.2f}")

        # 랜덤 색상 생성
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # 2D 이미지에 바운딩 박스 및 라벨 추가
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)  # 텍스트 크기 키움

    # 결과 이미지 반환
    return img
