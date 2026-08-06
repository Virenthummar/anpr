import cv2
import os
import argparse
from frame_enhancer import enhance_frame

def main():
    parser = argparse.ArgumentParser(description="Test Image Pre-processing & Frame Enhancer Engine")
    parser.add_argument("--image", default="num.png", help="Path to input image file")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not read image {args.image}")
        return

    print("\n" + "=" * 60)
    print("      RUNNING IMAGE FRAME ENHANCEMENT PIPELINE TEST")
    print("=" * 60)

    # 1. Test standard enhancement & QA logging
    print("\n---> Test 1: Standard Frame Enhancement (with QA review logging)")
    enhanced_std = enhance_frame(frame, log_qa=True, save_name="standard")

    # 2. Test Low-Light Simulation
    print("\n---> Test 2: Low-Light Night-Vision Simulation")
    dark_frame = (frame * 0.25).astype("uint8")  # Darken frame to 25% brightness
    enhanced_dark = enhance_frame(dark_frame, log_qa=True, save_name="low_light_sim")

    # 3. Test Motion Blur Simulation
    print("\n---> Test 3: Motion-Blur Vehicle Simulation")
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 5)  # Add heavy motion blur
    enhanced_blur = enhance_frame(blurred_frame, log_qa=True, save_name="motion_blur_sim")

    print("\nAll enhancement tests completed successfully! Check the 'qa_logs/' folder for side-by-side QA comparison images.")

if __name__ == "__main__":
    main()
