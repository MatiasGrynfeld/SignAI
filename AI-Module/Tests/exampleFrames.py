import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_folder, num_frames=9):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_indices = list(range(0, num_frames))
    middle_start = total_frames // 2 - num_frames // 2
    middle_indices = list(range(middle_start, middle_start + num_frames))
    end_start = total_frames - num_frames
    end_indices = list(range(end_start, total_frames))
    
    start_folder = os.path.join(output_folder, 'start')
    middle_folder = os.path.join(output_folder, 'middle')
    end_folder = os.path.join(output_folder, 'end')
    os.makedirs(start_folder, exist_ok=True)
    os.makedirs(middle_folder, exist_ok=True)
    os.makedirs(end_folder, exist_ok=True)
    
    for idx in start_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(start_folder, f"frame_{idx}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
        else:
            print(f"Error: Could not read frame {idx}")
    
    for idx in middle_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(middle_folder, f"frame_{idx}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
        else:
            print(f"Error: Could not read frame {idx}")
    
    for idx in end_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(end_folder, f"frame_{idx}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
        else:
            print(f"Error: Could not read frame {idx}")
    
    # Release the video capture object
    cap.release()

# Example usage
video_path = Path(os.getcwd()) / 'AI-Module' / 'Resources' / 'Videos' / '18YQlS_nliI-8-rgb_front.mp4'
output_folder = str(Path(os.getcwd()) / 'AI-Module' / 'Resources' / 'Videos' / 'Frames')
extract_frames(video_path, output_folder)