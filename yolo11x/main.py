import base64
import io
import json
from PIL import Image
from ultralytics import YOLO

def init_context(context):
    context.logger.info("Init context...  0%")
    model = YOLO("yolo11x.pt")
    context.user_data.model = model
    context.logger.info("Init context... 100%")

def handler(context, event):
    context.logger.info("Run YOLO11x model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = Image.open(buf).convert("RGB")
    results = context.user_data.model.predict(image, conf=threshold)
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = results[0].names[cls]
        detections.append({
            "confidence": str(conf),
            "label": label,
            "points": [x1, y1, x2, y2],
            "type": "rectangle",
            "attributes": []
        })
    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200
    )
