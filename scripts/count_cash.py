#!/usr/bin/env python3
"""
Cash counting using RT-DETR (Real-Time Detection Transformer).

Uses Ultralytics RT-DETR to detect and count cash/bills in a video.
"""

import argparse
import logging
from pathlib import Path
from collections import Counter

from ultralytics import RTDETR
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def count_cash_in_video(
    video_path: str,
    model_name: str = "rtdetr-l",
    output_path: str = None,
    conf_threshold: float = 0.3,
    target_classes: list = None,  # e.g., ["bill", "money", "cash"]
    show: bool = False,
):
    """
    Count cash/objects in a video using RT-DETR.
    
    Args:
        video_path: Path to input video
        model_name: RT-DETR model variant (rtdetr-l, rtdetr-x)
        output_path: Path to save annotated video (optional)
        conf_threshold: Confidence threshold
        target_classes: Specific class names to count (None = all)
        show: Show video while processing
        
    Returns:
        dict with counts per class and per frame
    """
    # Load RT-DETR model
    logger.info(f"Loading RT-DETR model: {model_name}")
    model = RTDETR(f"{model_name}.pt")
    
    # Get class names
    class_names = model.names
    logger.info(f"Model has {len(class_names)} classes")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Setup video writer if output path specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_counts = []
    total_counts = Counter()
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run RT-DETR inference
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Count detections in this frame
        frame_counter = Counter()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = class_names[cls_id]
                conf = float(box.conf[0])
                
                # Filter by target classes if specified
                if target_classes is None or cls_name.lower() in [t.lower() for t in target_classes]:
                    frame_counter[cls_name] += 1
                    total_counts[cls_name] += 1
        
        frame_counts.append({
            "frame": frame_idx,
            "counts": dict(frame_counter),
            "total": sum(frame_counter.values()),
        })
        
        # Draw annotations
        if writer or show:
            annotated = results[0].plot()
            
            # Add count overlay
            y_offset = 30
            for cls_name, count in frame_counter.most_common(5):
                text = f"{cls_name}: {count}"
                cv2.putText(annotated, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y_offset += 35
            
            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow("RT-DETR Cash Counter", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            logger.info(f"Processed {frame_idx}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    
    # Find max counts (peak detection)
    max_frame = max(frame_counts, key=lambda x: x["total"])
    
    result = {
        "video_path": video_path,
        "model": model_name,
        "total_frames": frame_idx,
        "fps": fps,
        "class_counts": dict(total_counts),
        "max_objects_in_frame": max_frame,
        "avg_objects_per_frame": sum(f["total"] for f in frame_counts) / max(1, len(frame_counts)),
    }
    
    logger.info(f"\n=== Results ===")
    logger.info(f"Total detections by class: {dict(total_counts)}")
    logger.info(f"Peak frame: {max_frame['frame']} with {max_frame['total']} objects")
    logger.info(f"Average objects per frame: {result['avg_objects_per_frame']:.1f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Count cash using RT-DETR')
    parser.add_argument('video', type=str, help='Path to input video')
    parser.add_argument('--model', type=str, default='rtdetr-l', 
                        choices=['rtdetr-l', 'rtdetr-x'],
                        help='RT-DETR model variant')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save annotated video')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold')
    parser.add_argument('--show', action='store_true',
                        help='Show video while processing')
    
    args = parser.parse_args()
    
    result = count_cash_in_video(
        video_path=args.video,
        model_name=args.model,
        output_path=args.output,
        conf_threshold=args.conf,
        show=args.show,
    )
    
    print(f"\nFinal Results: {result}")


if __name__ == '__main__':
    main()
