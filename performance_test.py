#!/usr/bin/env python3
"""
Performance testing script for Face Capture 2.0
Measures FPS and identifies bottlenecks in the pipeline
"""

import time
import cv2
import numpy as np
import mss
from pathlib import Path
from detection.yolo_detector import YOLODetector
from segmentation.face_segmentor import FaceSegmentor

def test_yolo_performance(detector, num_frames=100):
    """Test YOLOv8 detection performance"""
    print("🔍 Testing YOLO Detection Performance...")
    
    # Create test frame (512x512 like your crop)
    test_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    start_time = time.time()
    for i in range(num_frames):
        detection = detector.detect(test_frame)
    end_time = time.time()
    
    fps = num_frames / (end_time - start_time)
    print(f"   YOLO FPS: {fps:.1f}")
    print(f"   Time per detection: {1000/fps:.1f}ms")
    return fps

def test_face_segmentation_performance(segmentor, num_frames=100):
    """Test MediaPipe face segmentation performance"""
    print("👤 Testing Face Segmentation Performance...")
    
    # Create test frame (1280x720 like your webcam)
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    start_time = time.time()
    for i in range(num_frames):
        _, masked = segmentor.process_frame(test_frame)
    end_time = time.time()
    
    fps = num_frames / (end_time - start_time)
    print(f"   Face Segmentation FPS: {fps:.1f}")
    print(f"   Time per segmentation: {1000/fps:.1f}ms")
    return fps

def test_screen_capture_performance(num_frames=100):
    """Test screen capture performance"""
    print("🖥️  Testing Screen Capture Performance...")
    
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        
        # Test full screen capture
        start_time = time.time()
        for i in range(num_frames):
            screen = np.array(sct.grab(monitor))
            screen_bgr = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        end_time = time.time()
        
        full_fps = num_frames / (end_time - start_time)
        print(f"   Full Screen Capture FPS: {full_fps:.1f}")
        
        # Test cropped capture (optimized version)
        crop_monitor = {
            "top": monitor["top"] + 200,
            "left": monitor["left"] + 700,
            "width": 512,
            "height": 512
        }
        
        start_time = time.time()
        for i in range(num_frames):
            screen_crop = np.array(sct.grab(crop_monitor))
            screen_bgr = cv2.cvtColor(screen_crop, cv2.COLOR_BGRA2BGR)
        end_time = time.time()
        
        crop_fps = num_frames / (end_time - start_time)
        print(f"   Cropped Screen Capture FPS: {crop_fps:.1f}")
        print(f"   Improvement: {crop_fps/full_fps:.1f}x faster")
        
        return full_fps, crop_fps

def test_overlay_performance(num_frames=1000):
    """Test RGBA overlay performance"""
    print("🎨 Testing Overlay Performance...")
    
    # Create test images
    base_img = np.full((1080, 1920, 3), (0, 255, 0), dtype=np.uint8)
    overlay_img = np.random.randint(0, 255, (200, 200, 4), dtype=np.uint8)
    position = (100, 100)
    
    # Import the optimized overlay function
    from real_time_overlay import overlay_rgba
    
    start_time = time.time()
    for i in range(num_frames):
        result = overlay_rgba(base_img.copy(), overlay_img, position)
    end_time = time.time()
    
    fps = num_frames / (end_time - start_time)
    print(f"   Overlay FPS: {fps:.1f}")
    print(f"   Time per overlay: {1000/fps:.2f}ms")
    return fps

def estimate_pipeline_performance():
    """Estimate overall pipeline performance based on component tests"""
    print("\n📊 Performance Analysis:")
    print("=" * 50)
    
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "runs/detect/train7/weights/best.pt"
    
    try:
        detector = YOLODetector(str(model_path), conf_threshold=0.5)
        segmentor = FaceSegmentor()
        
        # Test individual components
        yolo_fps = test_yolo_performance(detector, 50)  # Fewer frames for YOLO (it's slow)
        face_fps = test_face_segmentation_performance(segmentor, 100)
        full_screen_fps, crop_screen_fps = test_screen_capture_performance(200)
        overlay_fps = test_overlay_performance(1000)
        
        print("\n🎯 Bottleneck Analysis:")
        print("-" * 30)
        
        # Calculate theoretical max FPS based on slowest component
        components = {
            "YOLO Detection": yolo_fps,
            "Face Segmentation": face_fps / 6,  # Runs every 6th frame
            "Screen Capture": crop_screen_fps,
            "Overlay": overlay_fps
        }
        
        bottleneck = min(components, key=components.get)
        max_fps = components[bottleneck]
        
        print(f"Bottleneck: {bottleneck} ({components[bottleneck]:.1f} FPS)")
        print(f"Theoretical Max FPS: {max_fps:.1f}")
        
        # Optimizations applied
        print("\n⚡ Optimizations Applied:")
        print("-" * 30)
        print("✅ YOLO detection every 3rd frame (3x speedup)")
        print("✅ Face segmentation every 6th frame (1.5x speedup)")
        print("✅ Cropped screen capture (faster)")
        print("✅ Optimized RGBA overlay")
        print("✅ Reduced camera buffer size")
        print("✅ Faster MediaPipe model")
        
        # Calculate expected improvement
        optimized_yolo_fps = yolo_fps * 3  # Every 3rd frame
        optimized_face_fps = face_fps * 1.5  # Every 6th instead of 4th
        
        optimized_components = {
            "YOLO Detection": optimized_yolo_fps,
            "Face Segmentation": optimized_face_fps / 6,
            "Screen Capture": crop_screen_fps,
            "Overlay": overlay_fps
        }
        
        optimized_bottleneck = min(optimized_components, key=optimized_components.get)
        optimized_max_fps = optimized_components[optimized_bottleneck]
        
        print(f"\n🚀 Expected Performance:")
        print("-" * 30)
        print(f"New Bottleneck: {optimized_bottleneck}")
        print(f"Expected FPS: {optimized_max_fps:.1f}")
        print(f"Improvement: {optimized_max_fps/max_fps:.1f}x faster")
        
        if optimized_max_fps >= 30:
            print("🎉 Target: 30+ FPS - ACHIEVABLE!")
        elif optimized_max_fps >= 20:
            print("⚠️  Target: 20-30 FPS - Good performance")
        else:
            print("❌ Target: <20 FPS - May need further optimization")
            
        segmentor.release()
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Make sure your YOLO model exists at runs/detect/train7/weights/best.pt")

if __name__ == "__main__":
    print("🚀 Face Capture 2.0 - Performance Test")
    print("=" * 50)
    estimate_pipeline_performance()
