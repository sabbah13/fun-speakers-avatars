import cv2
import numpy as np
import argparse
import os
import re
import datetime
import random # Added for random frame selection
import difflib # Added for name matching
from collections import Counter
from deepface import DeepFace
from paddleocr import PaddleOCR
import base64
import subprocess # For running curl
import shlex # For safely splitting command strings
import json # For parsing curl JSON output
import sys # Added for sys.stderr
try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv library not found. Cannot load .env file.")
    print("Install using: pip install python-dotenv")
    load_dotenv = None # Set to None to handle checks later

try:
    from openai import OpenAI, APIError, RateLimitError
except ImportError:
    print("Error: OpenAI library not found. Please install it using: pip install openai")
    OpenAI = None # Set to None to handle checks later

# --- Initialize PaddleOCR ---
# Done globally to load model once.
print("Initializing PaddleOCR...")
try:
    ocr_reader = PaddleOCR(use_angle_cls=True, lang='cyrillic', show_log=False)
    print("PaddleOCR Initialized.")
except Exception as e:
    print(f"Error initializing PaddleOCR: {e}")
    print("Please ensure paddlepaddle and paddleocr are installed correctly.")
    exit()

# --- Configuration ---
# Face Detection Confidence (DeepFace doesn't expose easily, but we filter results)
MIN_FACE_CONFIDENCE = 0.1 # Adjust if needed, based on DeepFace output inspection

# Card Detection Parameters
MIN_RECT_WIDTH = 128
MIN_RECT_HEIGHT = 128
CARD_FACE_AREA_RATIO_MIN = 1.1 # Card area must be > 1.1x face area

# Pre-processing
MEDIAN_BLUR_KERNEL_SIZE = 3
DILATE_KERNEL = np.ones((3, 3), np.uint8)
DILATE_ITERATIONS = 1
CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150

# OCR ROI Configuration
OCR_ROI_HEIGHT_PERCENT = 0.10
OCR_HORIZONTAL_FILTER_RATIO = 1.5

# Debug Output Colors
FACE_COLOR = (0, 0, 255)   # Red
CARD_COLOR = (255, 0, 0)   # Blue
OCR_ROI_COLOR = (255, 255, 0) # Cyan
TEXT_COLOR = (0, 255, 255) # Yellow
BOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 1

# Avatar Generation Prompt (multiline string)
AVATAR_GENERATION_PROMPT = """
You are given multiple images of a person from a video conferencing platform. Each image shows a speaker's face or their avatar.

Your task is to:

1. Determine if the tile includes a visible human face or an avatar:
2. If a real face or avatar is visible, generate a cartoon avatar using the visual content of the tile.

Avatar Requirements:

- Aspect Ratio: 1:1 square.
- Style: 2D cartoon style with thick clean outlines, warm soft colors, and smooth shading.
- Expression: Friendly, positive, and approachable.
- Framing: Front-facing, upper body only.
- Clothing: Preserve and faithfully reproduce the outfit visible in the original tile.
- Facial Features: Must remain clearly recognizable as the original person.
- Exclusions: Do NOT include any microphones or headphones, even if visible in the source images. Do NOT include any hands.
- Enhancement: Subtly improve the appearance by making the person look:
  - Slightly younger (e.g., smoother skin, brighter eyes)
  - Slightly fitter (e.g., improved posture or subtle jawline tightening)
  - Generally more refreshed and polished
  - The enhancement must look natural and respect the person's identity.

- Background: Use a flat, soft color selected from this fixed palette. Choose the one that best matches or complements the tile's dominant background color:
  - Warm beige (#F5E8D3)
  - Soft mint green (#D2F5D3)
  - Dusty rose (#E8C1C1)
  - Pale lavender (#D9D3F5)
  - Light sky blue (#CDEBFA)
  - Soft coral (#F7C7B8)
  - Muted mustard (#F3E2B3)
  - Powder gray-blue (#D0D7E8)

Output:
Return only the generated enhanced cartoon avatar image.
"""

# Pricing constants (per 1 Million tokens or per image)
PRICE_GPT_IMAGE_1_INPUT_TEXT = 5.00 / 1_000_000
PRICE_GPT_IMAGE_1_INPUT_IMAGE = 10.00 / 1_000_000
PRICE_GPT_IMAGE_1_OUTPUT_TOKEN = 40.00 / 1_000_000 # Price per output image token

# --- Helper Functions ---

def calculate_cost(usage_data):
    """Calculates the estimated cost based on usage data (input/output tokens) and pricing."""
    input_text_tokens = 0
    input_image_tokens = 0
    output_tokens = 0

    if usage_data:
        output_tokens = usage_data.get('output_tokens', 0) or 0 # Get output tokens
        if usage_data.get('input_tokens_details'):
            details = usage_data['input_tokens_details']
            input_text_tokens = details.get('text_tokens', 0) or 0 # Ensure it's a number
            input_image_tokens = details.get('image_tokens', 0) or 0 # Ensure it's a number
        elif usage_data.get('input_tokens'):
             print("  Warning: input_tokens_details missing, cannot calculate separate text/image input costs accurately.", file=sys.stderr)

    cost_input_text = input_text_tokens * PRICE_GPT_IMAGE_1_INPUT_TEXT
    cost_input_image = input_image_tokens * PRICE_GPT_IMAGE_1_INPUT_IMAGE
    cost_output_tokens = output_tokens * PRICE_GPT_IMAGE_1_OUTPUT_TOKEN

    total_cost = cost_input_text + cost_input_image + cost_output_tokens

    return {
        "input_text_cost": cost_input_text,
        "input_image_cost": cost_input_image,
        "output_tokens_cost": cost_output_tokens,
        "total_cost": total_cost,
        "input_text_tokens": input_text_tokens,
        "input_image_tokens": input_image_tokens,
        "output_tokens": output_tokens,
    }

def setup_output_dirs(base_output_dir, debug_mode):
    """Creates timestamped output directories."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_path = os.path.join(base_output_dir, timestamp)
    debug_output_path = os.path.join(main_output_path, "debug") if debug_mode else None

    os.makedirs(main_output_path, exist_ok=True)
    if debug_output_path:
        os.makedirs(debug_output_path, exist_ok=True)

    print(f"Main output directory: {main_output_path}")
    if debug_output_path:
        print(f"Debug output directory: {debug_output_path}")

    return main_output_path, debug_output_path

def sanitize_foldername(name):
    """Removes characters invalid for folder names."""
    name = name.strip()
    name = name.replace(' ', '_')
    name = re.sub(r'[<>:"/\\|?*\']', '', name)
    # Limit length if necessary (optional)
    # name = name[:50]
    # Handle empty names after sanitization
    if not name:
        name = "_invalid_name_"
    return name

def detect_faces(frame):
    """Detects all faces in the frame."""
    print("Detecting faces...")
    try:
        # enforce_detection=False returns list even if no faces, or raises ValueError if detector fails
        face_results = DeepFace.extract_faces(img_path=frame, detector_backend='retinaface', enforce_detection=False, align=False)
        # Filter by confidence if possible (DeepFace structure might vary)
        valid_faces = [f for f in face_results if f.get('confidence', 0) >= MIN_FACE_CONFIDENCE]
        print(f"Detected {len(valid_faces)} face(s) meeting confidence.")
        return valid_faces
    except ValueError as ve:
         if "Face could not be detected" in str(ve):
             print("No faces detected by the model.")
             return []
         else:
              # Use simpler printing for the warning message
              print("Warning: Unexpected face detection ValueError: " + str(ve))
              return []
    except Exception as e:
        print(f"Error during face detection: {e}")
        return [] # Return empty list on other errors too

def preprocess_for_contours(frame):
    """Applies grayscale, blur, Canny, and dilation."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, MEDIAN_BLUR_KERNEL_SIZE)
    edges = cv2.Canny(blurred, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    dilated_edges = cv2.dilate(edges, DILATE_KERNEL, iterations=DILATE_ITERATIONS)
    return dilated_edges

def find_contours(edges):
    """Finds external contours."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"Found {len(contours)} raw contours.") # Less verbose
    return contours

def find_card_for_face(face_data, contours, frame):
    """Finds the smallest card contour containing the face center and meeting size criteria."""
    face_area_data = face_data['facial_area']
    fx, fy, fw, fh = face_area_data['x'], face_area_data['y'], face_area_data['w'], face_area_data['h']
    face_bbox_area = fw * fh
    face_cx = fx + fw // 2
    face_cy = fy + fh // 2

    candidate_cards = []
    for cnt in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        # 1. Contains face center?
        if not (x_c <= face_cx <= x_c + w_c and y_c <= face_cy <= y_c + h_c):
            continue
        # 2. Min dimensions?
        if not (w_c >= MIN_RECT_WIDTH and h_c >= MIN_RECT_HEIGHT):
            continue
        # 3. Larger than face?
        card_bbox_area = w_c * h_c
        if not (card_bbox_area > face_bbox_area * CARD_FACE_AREA_RATIO_MIN):
            continue

        # Passed all checks
        candidate_cards.append((card_bbox_area, (x_c, y_c, w_c, h_c)))

    if candidate_cards:
        # Select the one with the smallest area among valid candidates
        smallest_area, best_card_box = min(candidate_cards, key=lambda t: t[0])
        return best_card_box
    else:
        return None

def run_paddle_ocr_on_roi(image_crop):
    """Runs PaddleOCR on an image crop and returns filtered horizontal text."""
    if image_crop is None or image_crop.size == 0: return ""
    try:
        result = ocr_reader.ocr(image_crop, cls=True)
        if result and result[0]: # Check if result is not None and has content
            detected_texts = []
            for idx in range(len(result[0])):
                res = result[0][idx]
                bbox = res[0]
                text, conf = res[1]
                # Simple Horizontal Check
                x_coords = [p[0] for p in bbox]; y_coords = [p[1] for p in bbox]
                box_w = max(x_coords) - min(x_coords); box_h = max(y_coords) - min(y_coords)
                if box_w > box_h * OCR_HORIZONTAL_FILTER_RATIO:
                    detected_texts.append(text)
            return " ".join(detected_texts).strip()
        else:
            return ""
    except Exception as e:
        print(f"Error during PaddleOCR: {e}")
        return ""

def get_name_from_card_ocr(card_crop):
    """Extracts name from top/bottom ROI of a card crop using PaddleOCR."""
    if card_crop is None or card_crop.size == 0: return "Name not found"
    
    ch, cw = card_crop.shape[:2]
    roi_h = int(ch * OCR_ROI_HEIGHT_PERCENT)
    found_name = "Name not found"

    # Top ROI
    top_roi_crop = card_crop[0:roi_h, :]
    top_text = run_paddle_ocr_on_roi(top_roi_crop)
    if top_text:
        # print(f"    Found text in Top ROI: '{top_text}'")
        found_name = top_text

    # Bottom ROI (only if top failed)
    if found_name == "Name not found":
        bottom_roi_y1 = ch - roi_h
        bottom_roi_crop = card_crop[bottom_roi_y1:ch, :]
        bottom_text = run_paddle_ocr_on_roi(bottom_roi_crop)
        if bottom_text:
            # print(f"    Found text in Bottom ROI: '{bottom_text}'")
            found_name = bottom_text

    return found_name

# --- Avatar Generation Function (Using curl) ---

def generate_avatar_with_curl(person_folder_path, image_paths, api_key, n=1, quality="high"):
    """Generates avatar(s) using OpenAI API via curl command."""
    if not api_key:
        print("Error: OpenAI API key not provided. Cannot generate avatars.", file=sys.stderr)
        return
    if not image_paths:
        print(f"Skipping avatar generation for {os.path.basename(person_folder_path)}: No images found.")
        return

    print(f"\nGenerating avatar(s) for: {os.path.basename(person_folder_path)} using curl")
    print(f"Using {len(image_paths)} source image(s):")

    # --- Construct curl command ---
    api_url = "https://api.openai.com/v1/images/edits"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    command = [
        "curl", "-s",
        "-X", "POST", api_url,
        "-H", f"Authorization: {headers['Authorization']}",
        "-F", "model=gpt-image-1",
        "-F", f"prompt={AVATAR_GENERATION_PROMPT}",
        "-F", f"quality={quality}",
        "-F", f"n={n}",
        "-F", "size=1024x1024"
    ]
    for img_path in image_paths:
        if os.path.isfile(img_path):
             command.extend(["-F", f"image[]=@{img_path}"])
        else:
             print(f"  Warning: Image file not found, skipping: {img_path}", file=sys.stderr)
    if not any(part.startswith("image[]=@") for part in command):
        print("  Error: No valid image files found to send. Aborting.", file=sys.stderr)
        return

    # --- Execute curl command ---
    print(f"  Executing curl command (image paths hidden for brevity)...")
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)

        if process.stderr and ("Failed sending data" in process.stderr or "curl:" in process.stderr):
             print(f"  Error: curl execution failed.\n  stderr:\n{process.stderr}", file=sys.stderr)
             return
        if process.returncode != 0:
            print(f"  Error: curl command failed with return code {process.returncode}\n  stderr:\n{process.stderr}\n  stdout:\n{process.stdout}", file=sys.stderr)
            return

        # --- Process valid response ---
        print("  Curl command executed successfully. Processing JSON response...")
        try:
            response_data = json.loads(process.stdout)
            if "error" in response_data:
                 print(f"  Error: OpenAI API returned an error:\n    {response_data['error']}", file=sys.stderr)
                 return
            if "data" not in response_data or not isinstance(response_data["data"], list):
                 print(f"  Error: Unexpected JSON response format from API.\n  Response: {response_data}", file=sys.stderr)
                 return

            usage_data = response_data.get("usage")
            if usage_data:
                print("  API Usage Information:")
                print(f"    Total Tokens: {usage_data.get('total_tokens')}")
                print(f"    Output Tokens: {usage_data.get('output_tokens')}")
                if usage_data.get('input_tokens_details'):
                    details = usage_data['input_tokens_details']
                    print(f"    Input Details: Text={details.get('text_tokens', 'N/A')}, Image={details.get('image_tokens', 'N/A')}")
            else:
                print("  Warning: Usage data not found in API response.")

            saved_count = 0
            for i, img_data in enumerate(response_data["data"]):
                if "b64_json" in img_data:
                    b64_data = img_data["b64_json"]
                    img_bytes = base64.b64decode(b64_data)
                    # Save with the standard name in the main script
                    avatar_filename = os.path.join(person_folder_path, f"avatar_{i+1}.png")
                    print(f"  Decoding base64 data for avatar {i+1}...")
                    with open(avatar_filename, "wb") as f:
                        f.write(img_bytes)
                    print(f"  Saved avatar to: {avatar_filename}")
                    saved_count += 1
                else:
                    print(f"  Warning: No 'b64_json' data found for image {i+1} in response.", file=sys.stderr)
            if saved_count == 0:
                 print("  Warning: No avatars were successfully decoded and saved.", file=sys.stderr)

            if usage_data:
                 cost_info = calculate_cost(usage_data)
                 print("  Estimated Cost Calculation:")
                 print(f"    Input Text Tokens : {cost_info['input_text_tokens']:>8} -> ${cost_info['input_text_cost']:,.6f}")
                 print(f"    Input Image Tokens: {cost_info['input_image_tokens']:>8} -> ${cost_info['input_image_cost']:,.6f}")
                 print(f"    Output Tokens     : {cost_info['output_tokens']:>8} -> ${cost_info['output_tokens_cost']:,.6f}")
                 print(f"    -------------------------------------")
                 print(f"    Total Estimated Cost        -> ${cost_info['total_cost']:,.6f}")
            else:
                 print("  Could not calculate cost: Usage data missing.")

        except json.JSONDecodeError:
            print("  Error: Failed to decode JSON response from curl.", file=sys.stderr)
            print(f"  Raw stdout:\n{process.stdout}", file=sys.stderr)
        except base64.binascii.Error as b64e:
             print(f"  Error: Failed to decode base64 data: {b64e}", file=sys.stderr)
        except Exception as e:
             print(f"  Error: An unexpected error occurred during response processing: {type(e).__name__} - {e}", file=sys.stderr)

    except FileNotFoundError:
        print("  Error: 'curl' command not found. Please ensure curl is installed and in your PATH.", file=sys.stderr)
    except Exception as e:
        print(f"  Error: An unexpected error occurred while running subprocess: {type(e).__name__} - {e}", file=sys.stderr)

# --- Frame Processing Function ---

def process_frame(frame, frame_index, main_output_path, debug_output_path, input_basename, debug_mode, valid_faces, known_names, name_similarity_cutoff):
    """Processes a single frame: finds contours, cards, OCR, potentially matches known names, and saves outputs."""
    print(f" Processing frame {frame_index} with {len(valid_faces)} face(s)...")
    frame_height, frame_width = frame.shape[:2] # Get frame dimensions

    # Preprocess & Find Contours
    dilated_edges = preprocess_for_contours(frame)
    contours = find_contours(dilated_edges)
    print(f"  Found {len(contours)} contours.")

    # Associate Faces, Find Cards, Run OCR for this frame
    results = []
    for i, face_data in enumerate(valid_faces):
        face_box = face_data['facial_area']
        face_box_tuple = (face_box['x'], face_box['y'], face_box['w'], face_box['h'])
        print(f"  Processing Face {i} at {face_box_tuple[:2]}...")

        # Pass frame to find_card_for_face
        card_box = find_card_for_face(face_data, contours, frame)

        if card_box:
            print(f"   -> Found Card: {card_box}")
            cx, cy, cw, ch = card_box
            cy_end = min(cy + ch, frame.shape[0]) # Corrected cropping bounds check
            cx_end = min(cx + cw, frame.shape[1])
            card_crop = frame[cy:cy_end, cx:cx_end] # Use corrected bounds

            ocr_name = get_name_from_card_ocr(card_crop)
            print(f"   -> OCR Found Name: '{ocr_name}'")

            # --- Known Name Matching and Filtering Logic ---
            process_this_result = False
            final_name = "Name not found"
            is_known_name = False
            sanitized_folder_name = None # Initialize

            if known_names: # If known names list is provided
                if ocr_name != "Name not found":
                    matches = difflib.get_close_matches(ocr_name, known_names, n=1, cutoff=name_similarity_cutoff)
                    if matches:
                        final_name = matches[0] # Use the exact known name
                        is_known_name = True
                        sanitized_folder_name = sanitize_foldername(final_name) # Also create sanitized version for folder
                        process_this_result = True # Match found, process it
                        print(f"    ==> Matched OCR name '{ocr_name}' to known name '{final_name}' (cutoff: {name_similarity_cutoff})")
                    else:
                        # No good match found for this OCR name in the known list
                        print(f"    --> OCR name '{ocr_name}' did not match known names {known_names} above cutoff {name_similarity_cutoff}. Ignoring this card.")
                        # process_this_result remains False
                else:
                     # OCR failed to find a name, cannot match known names
                     print(f"    --> OCR could not find a name for Card {i}. Ignoring.")
                     # process_this_result remains False

            else: # If known names list is NOT provided
                process_this_result = True # Process all found cards
                final_name = ocr_name # Use original OCR name
                sanitized_folder_name = sanitize_foldername(final_name) # Sanitize for folder
                is_known_name = False # Mark as not known

            # Add to results only if it should be processed
            if process_this_result:
                 if final_name != "Name not found": # Only add if we actually have a name
                    # Store original name, sanitized name, and known status
                    results.append({'face_box': face_box_tuple, 'card_box': card_box, 
                                    'name': final_name, 'sanitized_name': sanitized_folder_name, 
                                    'is_known': is_known_name, 'face_index': i})
                 else:
                     # This case happens if --known-names was not provided, but OCR failed.
                     print(f"    --> OCR failed for Card {i} and no known names specified. Ignoring.")

        else: # if not card_box
            print("   -> No suitable card found for this face.")

    if not results:
        print(f"  No valid cards/names found (or matched to known names if specified) for any faces in frame {frame_index}.")
        return 0 # Return 0 saved cards

    # Generate Outputs for this frame (debug image, cropped cards)
    print(f"\n  --- Frame {frame_index} Filtered Results ({len(results)} associations) ---") # Log filtered count

    # --- Create Debug Image (if requested) ---
    if debug_mode and debug_output_path:
        print("  Generating debug image...")
        debug_frame = frame.copy()
        # Modified filename for video frames
        debug_filename = os.path.join(debug_output_path, f"debug_{input_basename}_frame-{frame_index}.png")

        # Draw all originally detected faces first for context, even if ignored later
        for i, face_data in enumerate(valid_faces):
             face_box = face_data['facial_area']
             fx, fy, fw, fh = face_box['x'], face_box['y'], face_box['w'], face_box['h']
             cv2.rectangle(debug_frame, (fx, fy), (fx + fw, fy + fh), FACE_COLOR, BOX_THICKNESS)
             cv2.putText(debug_frame, f"F{i}", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FACE_COLOR, FONT_THICKNESS)

        # Draw cards and names ONLY for successful associations in the filtered results
        for result in results:
            card_box = result.get('card_box')
            name = result.get('name', 'N/A')
            face_idx = result.get('face_index', '?') # Get original face index associated with this result

            if card_box: # Should always be true if it's in results now
                cx, cy, cw, ch = card_box
                cv2.rectangle(debug_frame, (cx, cy), (cx + cw, cy + ch), CARD_COLOR, BOX_THICKNESS)
                # ... (draw ROI rects) ...
                roi_h = int(ch * OCR_ROI_HEIGHT_PERCENT)
                if cy + roi_h <= frame.shape[0]:
                     cv2.rectangle(debug_frame, (cx, cy), (cx + cw, cy + roi_h), OCR_ROI_COLOR, 1)
                if cy + ch - roi_h >= 0:
                     cv2.rectangle(debug_frame, (cx, cy + ch - roi_h), (cx + cw, cy + ch), OCR_ROI_COLOR, 1)

                text_pos = (cx + 5, cy + ch - 10)
                if text_pos[1] > 0:
                     # Label with F{original_face_index}: Name
                     cv2.putText(debug_frame, f"F{face_idx}: {name}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        cv2.imwrite(debug_filename, debug_frame)
        print(f"  Saved debug image to: {debug_filename}")

    # --- Save Cropped Cards ---
    print("  Saving cropped cards...")
    saved_count = 0
    for result in results: # Use the filtered results
        card_box = result.get('card_box')
        name = result.get('name')
        face_idx = result.get('face_index')
        is_known = result.get('is_known')

        # This check is slightly redundant now as results are pre-filtered, but good practice
        if card_box and name and name != "Name not found":
            # Determine folder name: ALWAYS use the sanitized version
            folder_name = result.get('sanitized_name') # Get the pre-sanitized name
            if not folder_name:
                 # Fallback if sanitized_name wasn't stored for some reason (shouldn't happen)
                 folder_name = sanitize_foldername(name)
            
            name_dir = os.path.join(main_output_path, folder_name)
            os.makedirs(name_dir, exist_ok=True)

            # ... (cropping and saving logic remains the same) ...
            cx, cy, cw, ch = card_box
            cy_end = min(cy + ch, frame.shape[0])
            cx_end = min(cx + cw, frame.shape[1])
            card_crop_img = frame[cy:cy_end, cx:cx_end]
            base_frame_name = f"{input_basename}_frame-{frame_index}"
            crop_filename = os.path.join(name_dir, f"{base_frame_name}_card_{face_idx}.png")
            if card_crop_img.size > 0:
                cv2.imwrite(crop_filename, card_crop_img)
                print(f"   Saved card {face_idx} for '{name}' to: {crop_filename}")
                saved_count += 1
            else:
                 print(f"   Warning: Card crop {face_idx} for '{name}' was empty.")

    print(f"  Saved {saved_count} cropped cards for frame {frame_index}.")
    return saved_count


# --- Main Video Processing Function ---

def process_video(video_path, base_output_dir, num_frames_to_process, max_attempts_factor, debug_mode, known_names, name_similarity_cutoff):
    """Loads video, samples frames, detects faces, and processes N frames with faces."""

    # 1. Load Video
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found at {video_path}"); return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}"); return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Loaded video: {video_path} ({width}x{height} @ {fps:.2f} FPS, {total_frames} frames)")

    if total_frames == 0:
        print("Error: Video has zero frames.")
        cap.release()
        return

    # 2. Setup Output Dirs
    main_output_path, debug_output_path = setup_output_dirs(base_output_dir, debug_mode)
    input_basename = os.path.splitext(os.path.basename(video_path))[0]

    # 3. Frame Sampling and Processing Loop
    processed_frames_count = 0
    attempt_count = 0
    max_attempts = max_attempts_factor * num_frames_to_process
    processed_frame_indices = set() # Keep track of frames already checked/processed
    total_saved_cards = 0

    print(f"\nAttempting to process {num_frames_to_process} frames with faces (max attempts: {max_attempts})...")

    while processed_frames_count < num_frames_to_process and attempt_count < max_attempts:
        attempt_count += 1
        # Select a random frame index that hasn't been processed yet
        random_frame_index = -1
        temp_attempt = 0 # Avoid infinite loop if all frames checked
        # Try to find an *unused* random index. Max attempts to avoid infinite loop if few frames remain.
        while temp_attempt < total_frames * 2: # Safety break, e.g. 2x total frames checks
             idx = random.randint(0, total_frames - 1)
             if idx not in processed_frame_indices:
                 random_frame_index = idx
                 break
             temp_attempt +=1
        
        if random_frame_index == -1: # If couldn't find an unused frame after many tries
             print("Warning: Could not find an unprocessed frame index. Might be near end or unlucky sampling.")
             # Check if all frames were already tried
             if len(processed_frame_indices) >= total_frames:
                print("All frames have been checked.")
                break # Exit main loop if all frames checked
             else: # Try again in the next main loop iteration
                continue 

        print(f"\n--- Attempt {attempt_count}/{max_attempts} | Checking Frame {random_frame_index} ---")
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
        ret, frame = cap.read()

        if not ret:
            print(f"  Warning: Could not read frame {random_frame_index}. Skipping.")
            processed_frame_indices.add(random_frame_index) # Mark as checked anyway
            continue

        # Detect faces in the current frame
        valid_faces = detect_faces(frame.copy()) # Use copy for detection

        if valid_faces:
            print(f"  Faces found in frame {random_frame_index}. Processing...")
            # Pass known_names and cutoff down to process_frame
            saved_in_frame = process_frame(frame, random_frame_index, main_output_path, debug_output_path, input_basename, debug_mode, valid_faces, known_names, name_similarity_cutoff)
            total_saved_cards += saved_in_frame
            processed_frames_count += 1
            processed_frame_indices.add(random_frame_index) # Mark as processed (implies checked)
        else:
            print(f"  No faces found in frame {random_frame_index}. Skipping processing.")
            processed_frame_indices.add(random_frame_index) # Mark as checked


    # 4. Cleanup and Summary
    print("\n--- Processing Complete ---")
    if processed_frames_count == num_frames_to_process:
        print(f"Successfully processed {processed_frames_count} frames with faces.")
    elif attempt_count >= max_attempts:
        print(f"Reached maximum attempts ({max_attempts}). Processed {processed_frames_count}/{num_frames_to_process} frames with faces.")
    else:
        # Could happen if all frames checked before reaching num_frames_to_process
        print(f"Finished processing. Found and processed {processed_frames_count} frames with faces out of {len(processed_frame_indices)} unique frames checked.")
    print(f"Total cropped cards saved: {total_saved_cards}")

    cap.release()
    print("Video released.")
    return main_output_path # Return the path for avatar generation


# --- Script Entry Point ---

if __name__ == "__main__":
    # Load environment variables from .env file first
    if load_dotenv:
        print("Loading environment variables from .env file...")
        # Make loading explicit, verbose, and overriding
        dotenv_loaded = load_dotenv(dotenv_path='.env', verbose=True, override=True)
        if not dotenv_loaded:
            print("Warning: .env file not found in the current directory.", file=sys.stderr)
    else:
        print("Skipping .env file loading (python-dotenv not found).")

    parser = argparse.ArgumentParser(description="Detect faces/cards, extract names, optionally generate avatars.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output", type=str, default=".", help="Base directory for timestamped output folder (default: current directory).")
    parser.add_argument("-n", "--num-frames", type=int, default=5, help="Number of random frames with faces to process (default: 5).")
    parser.add_argument("--max-attempts-factor", type=int, default=10, help="Factor to determine max random frame checks (max_attempts = factor * num_frames, default: 10).")
    parser.add_argument("--debug", action='store_true', help="Enable saving of debug images with annotations.")
    parser.add_argument('--known-names', nargs='+', help='Optional list of known names to match against OCR results.')
    parser.add_argument('--name-similarity-cutoff', type=float, default=0.7, help='Similarity cutoff for matching OCR results to known names (0.0 to 1.0, default: 0.7).')
    parser.add_argument("--generate-avatar", action='store_true', help="Enable generation of cartoon avatars using OpenAI.")
    parser.add_argument("--avatar-n", type=int, default=1, help="Number of avatars to generate per person (default: 1).")
    parser.add_argument("--avatar-quality", type=str, default="high", choices=['low', 'medium', 'high'], help="Quality of generated avatars (default: high).")


    args = parser.parse_args()

    # Determine API Key (Keep using .env or environment variable)
    api_key = os.environ.get("OPENAI_API_KEY")

    # --- Input Validation ---
    if not os.path.isdir(args.output):
        print(f"Error: Output directory '{args.output}' not found. Please create it.")
        exit()
    if not os.path.isfile(args.input):
         print(f"Error: Input video file '{args.input}' not found.")
         exit()
    if args.generate_avatar and not OpenAI:
         print("Error: --generate-avatar requested, but OpenAI library is not installed or failed to import.")
         exit()
    if args.generate_avatar and not api_key:
         print("Error: --generate-avatar requested, but OpenAI API key is missing. Provide --openai-api-key or set OPENAI_API_KEY environment variable.")
         exit()

    # --- Run Video Processing ---
    timestamped_output_path = None # Initialize
    try:
        # Get the specific timestamped output path
        timestamped_output_path = process_video(args.input, args.output, args.num_frames, args.max_attempts_factor, args.debug, args.known_names, args.name_similarity_cutoff)

    except Exception as e:
        print(f"\n--- An unexpected error occurred during video processing: {e} ---")
        # Decide if we should exit or try avatar generation anyway (likely exit)
        exit() # Exit if video processing fails

    # --- Run Avatar Generation (if enabled and video processing succeeded) ---
    # This block now correctly runs only *after* process_video is finished
    if args.generate_avatar and timestamped_output_path and os.path.isdir(timestamped_output_path):
        print("\n--- Starting Avatar Generation ---")
        if not api_key: # Double check key just before generation
             print("Error: API key unavailable. Cannot generate avatars.")
        else:
            person_folders = [os.path.join(timestamped_output_path, d) for d in os.listdir(timestamped_output_path) if os.path.isdir(os.path.join(timestamped_output_path, d)) and d != 'debug']

            if not person_folders:
                print("No person-specific folders found in output directory. Cannot generate avatars.")
            else:
                for person_dir in person_folders:
                    # Find all .png images (the cropped cards) in the person's directory
                    card_images = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.lower().endswith('.png') and not f.startswith('avatar_')] # Exclude previous avatars
                    if card_images:
                         # Pass quality argument
                         generate_avatar_with_curl(person_dir, card_images, api_key, args.avatar_n, args.avatar_quality)
                    else:
                         print(f"No card images found in {person_dir}, skipping avatar generation.")
        print("--- Avatar Generation Finished ---")

    # --- Removed the redundant call to process_video here --- 