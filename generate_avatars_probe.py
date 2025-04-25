import os
import argparse
import base64
import sys
import subprocess # For running curl
import shlex # For safely splitting command strings
import json # For parsing curl JSON output

try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv library not found. Cannot load .env file.", file=sys.stderr)
    print("Install using: pip install python-dotenv", file=sys.stderr)
    load_dotenv = None

# OpenAI library import removed as we are using curl now
# try:
#     from openai import OpenAI, APIError, RateLimitError
# except ImportError:
#     print("Error: OpenAI library not found. Please install it using: pip install openai", file=sys.stderr)
#     OpenAI = None

# Avatar Generation Prompt (copied from detect_card.py)
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
- Exclusions: Do NOT include any microphones or headphones, even if visible in the source images.
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

# Pricing constants (per 1 Million tokens)
PRICE_GPT_IMAGE_1_INPUT_TEXT = 5.00 / 1_000_000
PRICE_GPT_IMAGE_1_INPUT_IMAGE = 10.00 / 1_000_000
PRICE_GPT_IMAGE_1_OUTPUT_TOKEN = 40.00 / 1_000_000 # NEW: Price per output image token

def calculate_cost(usage_data): # Removed num_images_generated
    """Calculates the estimated cost based on usage data (input/output tokens) and pricing."""
    input_text_tokens = 0
    input_image_tokens = 0
    output_tokens = 0 # NEW: Get output tokens

    if usage_data:
        output_tokens = usage_data.get('output_tokens', 0) or 0 # Get output tokens
        if usage_data.get('input_tokens_details'):
            details = usage_data['input_tokens_details']
            input_text_tokens = details.get('text_tokens', 0) or 0 # Ensure it's a number
            input_image_tokens = details.get('image_tokens', 0) or 0 # Ensure it's a number
        # Fallback if input_tokens_details is missing but input_tokens exists
        elif usage_data.get('input_tokens'):
             # Can't split text/image, maybe log a warning or assume all are image?
             # For now, let's assume it might represent the sum if details missing
             # Or default to 0 if we can't be sure. Let's default to 0 for text/image split.
             print("  Warning: input_tokens_details missing, cannot calculate separate text/image input costs accurately.", file=sys.stderr)


    cost_input_text = input_text_tokens * PRICE_GPT_IMAGE_1_INPUT_TEXT
    cost_input_image = input_image_tokens * PRICE_GPT_IMAGE_1_INPUT_IMAGE
    cost_output_tokens = output_tokens * PRICE_GPT_IMAGE_1_OUTPUT_TOKEN # NEW: Calculate output cost based on tokens

    total_cost = cost_input_text + cost_input_image + cost_output_tokens # Adjusted total

    return {
        "input_text_cost": cost_input_text,
        "input_image_cost": cost_input_image,
        "output_tokens_cost": cost_output_tokens, # Renamed for clarity
        "total_cost": total_cost,
        "input_text_tokens": input_text_tokens,
        "input_image_tokens": input_image_tokens,
        "output_tokens": output_tokens, # Include output tokens
    }

def generate_avatar_with_curl(person_folder_path, image_paths, api_key, n=1):
    """Generates avatar(s) using OpenAI API via curl command."""
    if not api_key:
        print("Error: OpenAI API key not provided. Cannot generate avatars.", file=sys.stderr)
        return
    if not image_paths:
        print(f"Skipping avatar generation for {os.path.basename(person_folder_path)}: No images found.")
        return

    print(f"\nGenerating avatar(s) for: {os.path.basename(person_folder_path)} using curl")
    print(f"Using {len(image_paths)} source image(s):")
    # for img_p in image_paths:
    #     print(f"  - {img_p}") # Keep log shorter

    # --- Construct curl command ---
    api_url = "https://api.openai.com/v1/images/edits"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    # Base command parts
    command = [
        "curl", "-s", # -s for silent mode (don't show progress)
        "-X", "POST", api_url,
        "-H", f"Authorization: {headers['Authorization']}",
        "-F", "model=gpt-image-1",
        "-F", f"prompt={AVATAR_GENERATION_PROMPT}",
        "-F", f"quality=high",
        "-F", f"n={n}",
        "-F", "size=1024x1024"
        # response_format is implicit b64_json for gpt-image-1 edit
    ]

    # Add image files
    for img_path in image_paths:
        # Basic check if file exists before adding
        if os.path.isfile(img_path):
             command.extend(["-F", f"image[]=@{img_path}"])
        else:
             print(f"  Warning: Image file not found, skipping: {img_path}", file=sys.stderr)

    # Check if any images were actually added
    if not any(part.startswith("image[]=@") for part in command):
        print("  Error: No valid image files found to send. Aborting.", file=sys.stderr)
        return

    # --- Execute curl command ---
    print(f"  Executing curl command (image paths hidden for brevity)...")
    # print(" ".join(shlex.quote(part) for part in command)) # For debugging the exact command

    try:
        # Run curl and capture stdout (JSON response) and stderr
        process = subprocess.run(command, capture_output=True, text=True, check=False) # Don't check=True yet

        # Check stderr for curl errors first
        if process.stderr:
            # Filter out potential non-error curl messages if needed
            if "Failed sending data" in process.stderr or "curl:" in process.stderr:
                 print(f"  Error: curl execution failed.", file=sys.stderr)
                 print(f"  stderr:\n{process.stderr}", file=sys.stderr)
                 return
            # else: # Might be informational messages from curl like progress, ignore if -s used
            #    pass

        # Check return code
        if process.returncode != 0:
            print(f"  Error: curl command failed with return code {process.returncode}", file=sys.stderr)
            print(f"  stderr:\n{process.stderr}", file=sys.stderr)
            print(f"  stdout:\n{process.stdout}", file=sys.stderr)
            return

        # --- Process valid response ---
        print("  Curl command executed successfully. Processing JSON response...")
        try:
            response_data = json.loads(process.stdout)
            # Check for API error within the JSON response
            if "error" in response_data:
                 print(f"  Error: OpenAI API returned an error:", file=sys.stderr)
                 print(f"    {response_data['error']}", file=sys.stderr)
                 return

            if "data" not in response_data or not isinstance(response_data["data"], list):
                 print(f"  Error: Unexpected JSON response format from API.", file=sys.stderr)
                 print(f"  Response: {response_data}", file=sys.stderr)
                 return

            # --- Log Usage Data ---
            usage_data = response_data.get("usage")
            if usage_data:
                print("  API Usage Information:")
                print(f"    Total Tokens: {usage_data.get('total_tokens')}")
                if usage_data.get('input_tokens_details'):
                    details = usage_data['input_tokens_details']
                    print(f"    Input Details: Text={details.get('text_tokens', 'N/A')}, Image={details.get('image_tokens', 'N/A')}")
            else:
                print("  Warning: Usage data not found in API response.")

            # Save the generated image(s)
            saved_count = 0
            num_generated = len(response_data.get("data", []))
            for i, img_data in enumerate(response_data["data"]):
                if "b64_json" in img_data:
                    b64_data = img_data["b64_json"]
                    img_bytes = base64.b64decode(b64_data)
                    avatar_filename = os.path.join(person_folder_path, f"avatar_probe_curl_{i+1}.png")
                    print(f"  Decoding base64 data for avatar {i+1}...")
                    with open(avatar_filename, "wb") as f:
                        f.write(img_bytes)
                    print(f"  Saved probe avatar to: {avatar_filename}")
                    saved_count += 1
                else:
                    print(f"  Warning: No 'b64_json' data found for image {i+1} in response.", file=sys.stderr)

            if saved_count == 0:
                 print("  Warning: No avatars were successfully decoded and saved.", file=sys.stderr)

            # --- Calculate and Print Cost (Using Token-Based Output Cost) ---
            if usage_data:
                 cost_info = calculate_cost(usage_data) # Pass only usage data now
                 print("  Estimated Cost Calculation:")
                 print(f"    Input Text Tokens : {cost_info['input_text_tokens']:>8} -> ${cost_info['input_text_cost']:,.6f}")
                 print(f"    Input Image Tokens: {cost_info['input_image_tokens']:>8} -> ${cost_info['input_image_cost']:,.6f}")
                 print(f"    Output Tokens     : {cost_info['output_tokens']:>8} -> ${cost_info['output_tokens_cost']:,.6f}") # Changed label and value
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


def main():
    # Load environment variables from .env file first
    if load_dotenv:
        print("Loading environment variables from .env file...")
        # Make loading explicit, verbose, and overriding
        dotenv_loaded = load_dotenv(dotenv_path='.env', verbose=True, override=True)
        if not dotenv_loaded:
            print("Warning: .env file not found in the current directory.", file=sys.stderr)
    else:
        print("Skipping .env file loading (python-dotenv not found).")

    parser = argparse.ArgumentParser(description="Generate avatars from pre-existing card image folders using curl.")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to the timestamped output directory containing person folders.")
    parser.add_argument("--avatar-n", type=int, default=1, help="Number of avatars to generate per person (default: 1).")

    args = parser.parse_args()

    # Determine API Key
    api_key = os.environ.get("OPENAI_API_KEY")
    #  # api_key print removed from history# REMOVED for security

    # --- Input Validation ---
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found or is not a directory.", file=sys.stderr)
        exit(1)
    # No longer need to check for OpenAI library
    if not api_key:
         print("Error: OpenAI API key is missing. Set OPENAI_API_KEY environment variable in your .env file.", file=sys.stderr)
         exit(1)

    # --- Run Avatar Generation ---
    print(f"\n--- Starting Avatar Generation Probe (curl) for directory: {args.input_dir} ---")

    try:
        person_folders = [os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d)) and d.lower() != 'debug']

        if not person_folders:
            print("No person-specific subfolders found in input directory.")
        else:
            print(f"Found person folders: {[os.path.basename(p) for p in person_folders]}")
            for person_dir in person_folders:
                # Find all .png images (the cropped cards) in the person's directory
                card_images = [os.path.join(person_dir, f) for f in os.listdir(person_dir)
                               if f.lower().endswith('.png') and not f.lower().startswith('avatar_')]

                if card_images:
                     card_images.sort() # Simple sort
                     generate_avatar_with_curl(person_dir, card_images, api_key, args.avatar_n) # Use new function
                else:
                     print(f"No card images (.png) found in {person_dir}, skipping avatar generation.")
        print("--- Avatar Generation Probe (curl) Finished ---")

    except Exception as e:
        print(f"\n--- An unexpected error occurred during main execution: {e} ---", file=sys.stderr)


if __name__ == "__main__":
    main() 