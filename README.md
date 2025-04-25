# Fun Speakers Avatars

This project analyzes video recordings of online meetings (e.g., Zoom, Google Meet) to identify speakers, extract their names from name cards/tiles, save images of their tiles, and optionally generate cartoon-style avatars based on those images using the OpenAI API.

![Example Avatar Generation](assets/images/demo.png) <!-- Add your example image here -->

## New 🚀

This project now leverages the **latest OpenAI Image Generation model (`gpt-image-1`)**, available via the Edits API! This allows for high-quality, customized avatar generation based directly on speaker images from video calls.

## Features

*   Processes video files (e.g., MP4).
*   Samples random frames from the video.
*   Detects faces in frames using `DeepFace`.
*   Identifies potential speaker "cards" (rectangular tiles containing faces) using OpenCV contour detection.
*   Extracts names displayed on cards using `PaddleOCR` (supports Cyrillic and English/Latin alphabets).
*   Optionally matches OCR-extracted names against a provided list of known participants.
*   Saves cropped images of speaker cards into timestamped directories, organized by speaker name.
*   Optionally generates cartoon-style avatars using the OpenAI Image API (`gpt-image-1` model via `curl`) based on the extracted card images.
*   Creates debug images showing detected faces, cards, and OCR ROIs.
*   Handles `.env` file for API key management.

## Prerequisites

*   Python 3.x
*   `curl` command-line tool installed and available in your PATH.
*   Required Python libraries (install via `pip`):
    *   `opencv-python`
    *   `numpy`
    *   `deepface`
    *   `paddlepaddle` (or `paddlepaddle-gpu` if you have a compatible GPU)
    *   `paddleocr`
    *   `python-dotenv` (optional, for `.env` file support)
    *   `openai` (optional, if you plan to use the Python library instead of `curl` in the future)
*   An OpenAI API key with access to the `gpt-image-1` model (if using avatar generation). The organization associated with the key must be verified by OpenAI.

## Installation

1.  **Clone the repository:**
    ```bash
    # Replace with the actual URL after creating the GitHub repository
    git clone https://github.com/sabbah13/fun-speakers-avatars.git
    cd fun-speakers-avatars
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install opencv-python numpy deepface paddlepaddle paddleocr python-dotenv openai
    # Or use paddlepaddle-gpu if applicable
    # pip install paddlepaddle-gpu ...
    ```
    *Note: DeepFace and PaddleOCR might download model files on first run.*

3.  **(Optional) Create a `.env` file:**
    If using avatar generation, create a file named `.env` in the project root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```

## Quick Start

1.  Place your video file (e.g., `meeting.mp4`) in the project directory or provide the full path.
2.  Run the script with the input video path:

    ```bash
    python generate-avatars-from-video.py --input meeting.mp4
    ```

This will:
*   Process the video `meeting.mp4`.
*   Attempt to find and process 5 random frames containing faces.
*   Save cropped card images into a timestamped folder (e.g., `YYYYMMDD_HHMMSS/Speaker_Name/`) within the current directory (`.`).
*   It will *not* generate avatars by default.

## Usage

### Basic Usage

```bash
python generate-avatars-from-video.py --input <path_to_video> [--output <base_output_dir>]
```

*   `--input`: **Required.** Path to the input video file.
*   `--output`: Base directory where the timestamped results folder will be created. Defaults to the current directory (`.`).

### Extended Usage (All Options)

```bash
python generate-avatars-from-video.py \
    --input <path_to_video> \
    --output <base_output_dir> \
    -n <num_frames> \
    --max-attempts-factor <factor> \
    --debug \
    --known-names <Name1> <Name2> ... \
    --name-similarity-cutoff <cutoff> \
    --generate-avatar \
    --avatar-n <num_avatars> \
    --avatar-quality <quality>
```

**Parameters:**

*   `--input <path_to_video>`: **(Required)** Path to the input video file (e.g., `recordings/meeting.mp4`).
*   `--output <base_output_dir>`: Base directory for the timestamped output folder. Default: `.` (current directory).
*   `-n, --num-frames <num_frames>`: Number of random frames *with faces* to process. Default: `5`. The script will sample frames until it finds this many frames containing detectable faces.
*   `--max-attempts-factor <factor>`: Multiplier used to set the maximum number of random frames the script will check. `max_attempts = num_frames * factor`. Helps prevent infinite loops if the video has few frames with detectable faces. Default: `10`.
*   `--debug`: If set, saves annotated debug images in a `debug` subfolder within the timestamped output directory. Shows detected faces (red), cards (blue), and OCR regions (cyan).
*   `--known-names <Name1> <Name2> ...`: Provide a list of expected participant names (space-separated). If set, the script will only save cards where the OCR-detected name closely matches one of these known names (using fuzzy matching). Folder names will use the provided known name. Example: `--known-names "Alice Smith" "Bob Johnson"`.
*   `--name-similarity-cutoff <cutoff>`: Sets the threshold (0.0 to 1.0) for matching OCR names to `--known-names`. Higher values require a closer match. Default: `0.7`. Only used if `--known-names` is provided.
*   `--generate-avatar`: If set, enables the generation of cartoon avatars using the OpenAI API after processing the video. Requires a valid `OPENAI_API_KEY` in the environment or `.env` file.
*   `--avatar-n <num_avatars>`: Number of avatar variations to generate per person. Default: `1`. Only used if `--generate-avatar` is set.
*   `--avatar-quality <quality>`: Quality of the generated avatar images (`low`, `medium`, `high`). Default: `high`. Only used if `--generate-avatar` is set.

## Algorithm Explanation

1.  **Initialization:** Loads necessary libraries, initializes PaddleOCR (downloads models if needed, supports Cyrillic & English), and parses command-line arguments. Loads the OpenAI API key from `.env` or environment variables if present.
2.  **Video Loading:** Opens the specified video file using OpenCV.
3.  **Output Setup:** Creates a timestamped output directory (e.g., `20231027_103000`) and a `debug` subdirectory if `--debug` is enabled.
4.  **Frame Sampling Loop:**
    *   Randomly selects frame indices until the target number of frames *containing faces* (`--num-frames`) is processed or the maximum attempt limit (`num_frames * max_attempts_factor`) is reached.
    *   Avoids re-processing the same frame index.
5.  **Frame Processing (for each selected frame with faces):**
    *   **Face Detection:** Uses `DeepFace` (with `retinaface` backend) to detect all faces in the frame.
    *   **Image Preprocessing:** Converts the frame to grayscale, applies median blur, Canny edge detection, and dilation to enhance rectangular contours.
    *   **Contour Detection:** Finds all external contours in the preprocessed image using OpenCV.
    *   **Card Identification:** For each detected face:
        *   Iterates through the contours found.
        *   Identifies potential "cards" (speaker tiles) by checking if a contour's bounding box:
            *   Contains the center of the face.
            *   Meets minimum width and height requirements.
            *   Is significantly larger than the face bounding box.
        *   Selects the smallest valid contour bounding box as the card for that face.
    *   **OCR:**
        *   If a card is found for a face, crops the card region from the original frame.
        *   Defines small Regions of Interest (ROIs) at the top and bottom of the card crop (where names typically appear).
        *   Runs `PaddleOCR` on these ROIs to extract text. Filters for primarily horizontal text.
    *   **Name Matching (Optional):**
        *   If `--known-names` is provided, uses `difflib.get_close_matches` to compare the OCR result against the known names list.
        *   If a close enough match (above `--name-similarity-cutoff`) is found, the card is associated with the known name. Otherwise, the card is ignored for saving/avatar generation.
        *   If `--known-names` is *not* provided, the raw (sanitized) OCR result is used as the name.
    *   **Saving Cropped Cards:**
        *   If a valid name was found (either via OCR directly or matched to a known name), creates a subdirectory within the timestamped output folder using the sanitized name (e.g., `output/20231027_103000/Alice_Smith/`).
        *   Saves the cropped card image to this subdirectory (e.g., `video_frame-123_card_0.png`).
    *   **Debug Image Generation (Optional):** If `--debug` is set, draws rectangles for faces, cards, and OCR ROIs, adds name labels, and saves the annotated frame to the `debug` subdirectory.
6.  **Avatar Generation (Optional):**
    *   Executed *after* all selected video frames have been processed.
    *   If `--generate-avatar` is enabled and the API key is available:
        *   Iterates through each person's subdirectory in the output folder.
        *   Collects all saved card images (`.png`) for that person.
        *   Constructs and executes a `curl` command to call the OpenAI Edits API (`v1/images/edits`) using the `gpt-image-1` model.
        *   Sends the collected card images and the detailed avatar generation prompt.
        *   Receives the generated avatar image(s) as base64 encoded data.
        *   Decodes and saves the avatar(s) (e.g., `avatar_1.png`) into the corresponding person's folder.
        *   Calculates and prints the estimated cost based on API usage tokens.
7.  **Completion:** Prints a summary of processed frames and saved cards. Releases the video file.

## Acknowledgements

This project utilizes several fantastic open-source libraries:

*   **OpenCV:** For core computer vision tasks (reading video, image manipulation, contour detection). ([https://opencv.org/](https://opencv.org/))
*   **NumPy:** For efficient numerical operations on image data. ([https://numpy.org/](https://numpy.org/))
*   **DeepFace:** For robust face detection. ([https://github.com/serengil/deepface](https://github.com/serengil/deepface))
*   **PaddleOCR:** For accurate text recognition, including Cyrillic. ([https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR))
*   **python-dotenv:** For managing environment variables (like API keys). ([https://github.com/theskumar/python-dotenv](https://github.com/theskumar/python-dotenv))
*   **(Implied) OpenAI:** The OpenAI API is used for avatar generation via `curl`. ([https://openai.com/](https://openai.com/))

We thank the authors and contributors of these libraries for their invaluable work.

## TODO

*   [ ] Improve card detection logic (e.g., handle overlapping cards, use shape/color heuristics).
*   [ ] Add option to use different face detectors supported by DeepFace.
*   [ ] Implement batch processing for multiple videos.
*   [ ] Add configuration file for thresholds and parameters instead of command-line args only.
*   [ ] Explore alternative OCR engines.
*   [ ] Offer direct OpenAI Python library integration as an alternative to `curl`.
*   [ ] Add unit and integration tests.

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -am 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Create a new Pull Request.

Please ensure your code follows basic Python style guidelines and include updates to the README if necessary.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.