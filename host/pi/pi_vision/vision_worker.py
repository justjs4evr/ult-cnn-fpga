"""Official YOLOv5n ONNX + existing OpenCV DNN, without Torch on the Pi."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import queue
import re
import time

HERE = Path(__file__).resolve().parent
MODEL_SHA = '04f0e55c26f58d17145b36045780fe1250d5bd2187543e11568e5141d05b3262'


def class_names():
    text = (HERE / 'models/coco128.yaml').read_text(encoding='utf-8')
    values = dict((int(i), name.strip()) for i, name in re.findall(r'^  (\d+): (.+)$', text, re.M))
    if sorted(values) != list(range(80)):
        raise ValueError('Expected official 80-class COCO mapping')
    return [values[i] for i in range(80)]


class Detector:
    def __init__(self, threads=3):
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        cv2.setNumThreads(threads)
        model = HERE / 'models/yolov5n.onnx'
        if hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_SHA:
            raise ValueError('ONNX checksum mismatch')
        self.net = cv2.dnn.readNetFromONNX(str(model))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.names = class_names()

    def detect(self, image, confidence=.35):
        cv2, np = self.cv2, self.np
        started = time.monotonic()
        h, w = image.shape[:2]
        scale = min(640 / w, 640 / h)
        new_w, new_h = round(w * scale), round(h * scale)
        left, top = (640 - new_w) // 2, (640 - new_h) // 2
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[top:top+new_h, left:left+new_w] = cv2.resize(image, (new_w, new_h))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[-1] != 85:
            raise ValueError('Unexpected YOLOv5 output shape: ' + str(output.shape))
        rows = output[0].astype(np.float32)
        rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] >= confidence)]
        candidates = []
        if len(rows):
            labels = rows[:, 5:].argmax(axis=1)
            scores = rows[:, 4] * rows[np.arange(len(rows)), labels + 5]
            for row, label, score in zip(rows, labels, scores):
                if score < confidence:
                    continue
                cx, cy, bw, bh = row[:4]
                x1 = max(0., min(float(w), (cx - bw/2 - left) / scale))
                y1 = max(0., min(float(h), (cy - bh/2 - top) / scale))
                x2 = max(0., min(float(w), (cx + bw/2 - left) / scale))
                y2 = max(0., min(float(h), (cy + bh/2 - top) / scale))
                if x2 > x1 and y2 > y1:
                    candidates.append((int(label), float(score), [x1, y1, x2-x1, y2-y1]))
        detections = []
        for label in sorted({c[0] for c in candidates}):
            group = [c for c in candidates if c[0] == label]
            indices = cv2.dnn.NMSBoxes([c[2] for c in group], [c[1] for c in group], confidence, .45)
            for index in np.asarray(indices).reshape(-1):
                _, score, (x, y, bw, bh) = group[int(index)]
                detections.append(dict(class_id=label, name=self.names[label], confidence=score,
                    xyxy=[round(float(v), 2) for v in (x, y, x+bw, y+bh)]))
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return dict(detections=detections[:100], inference_and_postprocess_ms=(time.monotonic()-started)*1000,
                    input_size=640, original_size=[w, h], output_shape=list(output.shape))


def annotate(image, result, path):
    import cv2
    shown = image.copy()
    for d in result['detections']:
        x1, y1, x2, y2 = [int(x) for x in d['xyxy']]
        cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 220, 50), 2)
        cv2.putText(shown, f"{d['name']} {d['confidence']:.2f}", (x1, max(16, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 50), 1)
    if not cv2.imwrite(str(path), shown):
        raise OSError('Cannot write annotated image')


def publish(channel, value):
    if channel is None:
        return
    try:
        channel.put_nowait(value)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(value)
        except queue.Full:
            pass


def camera_worker(channel, stop, folder, max_frames=0):
    camera = None
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        from picamera2 import Picamera2
        detector = Detector()
        camera = Picamera2(0)
        # Picamera2 RGB888 produces BGR byte arrays, matching OpenCV convention.
        config = camera.create_video_configuration(main={'size': (640, 480), 'format': 'RGB888'},
            buffer_count=2, queue=False, controls={'FrameRate': 10})
        camera.configure(config)
        camera.start()
        time.sleep(1)
        count = 0
        with (folder / 'vision.jsonl').open('w', encoding='utf-8') as log:
            while not stop.is_set() and (not max_frames or count < max_frames):
                request = camera.capture_request()
                try:
                    image = request.make_array('main').copy()
                    metadata = request.get_metadata()
                finally:
                    request.release()
                captured = time.monotonic()
                sensor_ns = metadata.get('SensorTimestamp')
                sensor_age = ((time.clock_gettime(time.CLOCK_BOOTTIME)*1e9 - sensor_ns)/1e9
                              if sensor_ns is not None else None)
                result = detector.detect(image)
                count += 1
                result.update(status='ready', frame_id=count, captured_monotonic=captured,
                    completed_monotonic=time.monotonic(), sensor_timestamp_ns=sensor_ns,
                    sensor_age_at_capture_seconds=sensor_age, source='OV5647_CSI_camera_A',
                    mission_model_validated=False, traffic_light_color='not_implemented')
                log.write(json.dumps(result) + '\n')
                log.flush()
                publish(channel, result)
                if count <= 3:
                    annotate(image, result, folder / f'camera_A_{count:03d}.jpg')
        publish(channel, dict(status='finished', frames=count))
    except Exception as exc:
        value = dict(status='error', error=f'{type(exc).__name__}: {exc}')
        (folder / 'camera_error.json').write_text(json.dumps(value), encoding='utf-8')
        publish(channel, value)
    finally:
        if camera is not None:
            camera.stop()
            camera.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=Path)
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--output', type=Path, default=HERE / 'runs/camera_test')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.image:
        import cv2
        detector = Detector()
        image = cv2.imread(str(args.image))
        if image is None:
            raise ValueError('Cannot read image')
        result = detector.detect(image)
        (args.output / 'image_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        annotate(image, result, args.output / 'image_result.jpg')
        print(json.dumps(result))
    else:
        import threading
        camera_worker(None, threading.Event(), args.output, args.frames)
        if (args.output / 'camera_error.json').exists():
            raise SystemExit((args.output / 'camera_error.json').read_text())


if __name__ == '__main__':
    main()
