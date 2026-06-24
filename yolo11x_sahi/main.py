import base64
import io
import json
import numpy as np
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def init_context(context):
    context.logger.info("Init context...  0%")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path='yolo11n.pt',
        confidence_threshold=0.25,
        device="cpu"
    )
    context.user_data.detection_model = detection_model
    context.logger.info("Init context... 100%")

def handler(context, event):
    context.logger.info("Run YOLO11n with SAHI (Low RAM)")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf).convert("RGB")
    
    original_width, original_height = image.size
    
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        context.logger.info(f"Image resized to {image.size} to save RAM")
        
    resized_width, resized_height = image.size
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height
    
    image_np = np.array(image)
    
    threshold = float(data.get("threshold", 0.25))
    context.user_data.detection_model.confidence_threshold = threshold
    
    result = get_sliced_prediction(
        image_np,
        context.user_data.detection_model,
        slice_height=320,
        slice_width=320,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
        postprocess_match_metric='IOU',
        postprocess_match_threshold=0.5
    )
    
    detections = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox.to_coco_bbox()
        x1, y1, width, height = bbox
        x2 = x1 + width
        y2 = y1 + height
        
        x1 *= scale_x
        y1 *= scale_y
        x2 *= scale_x
        y2 *= scale_y
        
        detections.append({
            "confidence": str(pred.score.value),
            "label": pred.category.name,
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
