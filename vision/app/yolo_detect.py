from ultralytics import YOLO
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 연결할 수 없습니다.")
    exit()

model = YOLO(BASE_DIR/"yolo11s.pt")



# ============ Object Detection Function ============
def detect_objects() -> dict|None:
    '''
    카메라 프레임을 읽어 YOLO 추론을 합니다.
    ## return
    dict|None
    '''
    ret, frame = cap.read()

    if not ret:
        print("프레임을 가져올 수 없습니다.")
        return None
    

    result = model(frame, verbose=False)
    r = result[0]
    
    detected = []


    for box in r.boxes:
        conf = float(box.conf[0])
        if conf < 0.7: continue

        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        cls_name = r.names[cls_id]

        detected.append({
            "class": cls_name,
            "box": [x1,y1,x2,y2]
        })
    

    return detected



def realtime_cam():
    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임 읽기 실패")
            break 


        results = model(frame, verbose=False)
        r = results[0]


        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.7: continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {round(conf, 2)}", (x1,y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 2)

        cv2.imshow("camera", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


    
if __name__ == "__main__":
    print(detect_objects())