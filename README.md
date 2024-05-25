# object_detect
YOLOv8 모델을 사용하여 객체 탐지 및 길이 측정.
객체의 길이를 측정하고, 결과를 json으로 반환하는 Flask 서버.


## 24-05-25
Flask 서버 추가.
yolov8l 모델 사용.
Flask 및 분석 기능 분리.
RestAPI 요청으로 분석 요청 가능. (요청 주소: "http://127.0.0.1:5000/analyze")