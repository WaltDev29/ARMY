from ultralytics import YOLO
import cv2
from pathlib import Path
import atexit

BASE_DIR = Path(__file__).resolve().parent.parent

cap = None


model = YOLO(BASE_DIR/"yolo11s.pt")



def _get_cap() -> None:
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
    return cap

def _release_cap() -> None:
    global cap
    cap.release()
    print("cap released")

atexit.register(_release_cap)

# ============ Object Detection Function ============
def detect_objects() -> list[dict]|None:
    '''
    카메라 프레임을 읽어 YOLO 추론을 합니다.
    '''
    cap = _get_cap()
    if not cap.isOpened():
        print("카메라 연결 실패")
        return None
    
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
        x, y, w, h = map(float,box.xywh[0])
        cls_id = int(box.cls[0])
        cls_name = r.names[cls_id]

        detected.append({
            "class": cls_name,
            "box": [x1,y1,x2,y2],
            "xywh": [x, y, w, h]
        })
    

    return detected



def detect_objects_from_image(image: cv2.Mat) -> list[dict]|None:
    '''
    전달받은 이미지에 대해 YOLO 추론을 합니다.
    '''
    result = model(image, verbose=False)
    r = result[0]
    
    detected = []


    for box in r.boxes:
        conf = float(box.conf[0])
        if conf < 0.7: continue

        x1,y1,x2,y2 = map(int, box.xyxy[0])
        x, y, w, h = map(float,box.xywh[0])
        cls_id = int(box.cls[0])
        cls_name = r.names[cls_id]

        detected.append({
            "class": cls_name,
            "box": [x1,y1,x2,y2],
            "xywh": [x, y, w, h]
        })
    

    return detected



def realtime_cam() -> None:
    cap = _get_cap()
    if not cap.isOpened():
        print("카메라 연결 실패")
        return
    
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

    cv2.destroyAllWindows()


    
if __name__ == "__main__":
    print(detect_objects())