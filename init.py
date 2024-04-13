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