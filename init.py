import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow # google의 자체 개발 라이브러리에서 cv2_imshow를 import

model = YOLO("yolov8n.pt")

#results = model.predict("https://ultralytics.com/images/bus.jpg")
results = model.predict("images/1.jpg")
results2 = model.predict("images/2.jpg")
results3 = model.predict("images/3.jpg")

print(results[0])
print(results[0].boxes) # 좌상단 좌표, 우하단 좌표, confidence score, class id


# Extract bounding box dimensions
boxes = results[0].boxes.xywh.cpu()

for box in boxes:
    x, y, w, h = box
    print("Width of Box: {:.2f}, Height of Box: {:.2f}".format(w, h))

# xywh 에 tensor 형식으로 크기값이 들어있음.

# for i in range(len(results[0].names)) :
#     print(f"[{i}] : {results[0].names[i]}")


# while True:
    # ret, frame = cap.read()
    # results = model(frame, agnostic_nms=True)[0]

# if not results or len(results) == 0:
#     continue

boxes = []

for result in results:

    detection_count = result.boxes.shape[0]

    for i in range(detection_count):
        cls = int(result.boxes.cls[i].item())
        name = result.names[cls]
        confidence = float(result.boxes.conf[i].item())
        bounding_box = result.boxes.xyxy[i].cpu().numpy()

        # boxes에 인식한 객체 목록 저장
        boxes.append(result.boxes.xyxy[i].cpu().numpy())

        x = int(bounding_box[0])
        y = int(bounding_box[1])
        width = int(bounding_box[2] - x)
        height = int(bounding_box[3] - y)

        print(f"[{i}] {name} : {width}, {height}")

# https://stackoverflow.com/questions/76069484/obtaining-detected-object-names-using-yolov8

size_base_index = int(input("크기의 기준이 되는 물체의 번호를 입력하세요: "))

# x = int(boxes[size_base_index][0])
# y = int(boxes[size_base_index][1])
base_width = int(boxes[size_base_index][2] - x)
base_height = int(boxes[size_base_index][3] - y)

print(f"선택한 객체의 크기\n가로: {width} 세로: {height}")
base_cm_width = int(input("선택한 물체의 가로 길이를 cm단위로 입력해주세요: "))
base_cm_height = int(input("선택한 물체의 세로 길이를 cm단위로 입력해주세요: "))


for result in results:

    detection_count = result.boxes.shape[0]

    for i in range(detection_count):
        cls = int(result.boxes.cls[i].item())
        name = result.names[cls]
        confidence = float(result.boxes.conf[i].item())
        bounding_box = result.boxes.xyxy[i].cpu().numpy()

        # boxes에 인식한 객체 목록 저장
        boxes.append(result.boxes.xyxy[i].cpu().numpy())

        x = int(bounding_box[0])
        y = int(bounding_box[1])
        width = int(bounding_box[2] - x)
        height = int(bounding_box[3] - y)

        cm_width = width / base_width * base_cm_width
        cm_height = height / base_height * base_cm_height

        print(f"[{i}] {name} : {cm_width}cm, {cm_height}cm")



res_plotted = results[0].plot() # plot() 함수를 이용해서 이미지 내에 bounding box나 mask 등의 result 결과를 그릴 수 O
#cv2_imshow(res_plotted)
plt.imshow(res_plotted)
plt.axis('off')  # 축 눈금 숨기기
plt.show()

res_plotted2 = results2[0].plot()
res_plotted3 = results3[0].plot()

plt.imshow(res_plotted2)
plt.axis('off')  # 축 눈금 숨기기
plt.show()

plt.imshow(res_plotted3)
plt.axis('off')  # 축 눈금 숨기기
plt.show()