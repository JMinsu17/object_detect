# object_detect
YOLOv8 모델을 사용하여 객체 탐지 및 길이 측정.
길이 측정 결과를 json으로 저장하는 프로그램.


# 24-05-25
Flask 서버 추가.
yolov8l 모델 사용.
Flask 및 분석 기능 분리.
RestAPI 요청으로 분석 요청 가능. (요청 주소: "http://127.0.0.1:5000/analyze")