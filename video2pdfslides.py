import os
import time
import cv2
import imutils
import shutil
import img2pdf
import glob
import argparse
import numpy as np


############# Define constants
"""
To skip starting few frames: 
    1. set WARMUP manually for that video, 
    2. see a region which remains constant for few frames. and 
    set that in only_speaker_mode() function.
To remove the speaker region to go into background subtraction model:
see the speaker region by opening imshow window for that frame, and 
hardcode that manually in 
"""
OUTPUT_SLIDES_DIR = f"./output"
#orig
# FRAME_RATE = 3                   # no.of frames per second that needs to be processed, fewer the count faster the speed
# WARMUP = 0              # initial number of frames to be skipped
# FGBG_HISTORY = FRAME_RATE * 15   # no.of frames in background object
# VAR_THRESHOLD = 16               # Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model.
# DETECT_SHADOWS = False            # If true, the algorithm will detect shadows and mark them.
# MIN_PERCENT = 0.1                # min % of diff between foreground and background to detect if motion has stopped
# MAX_PERCENT = 3                  # max % of diff between foreground and background to detect if frame is still in motion

#new
FRAME_RATE = 3                   # no.of frames per second that needs to be processed, fewer the count faster the speed
WARMUP = 0              # initial number of frames to be skipped
FGBG_HISTORY = FRAME_RATE * 15   # no.of frames in background object
VAR_THRESHOLD = 16               # Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model.
DETECT_SHADOWS = False            # If true, the algorithm will detect shadows and mark them.
MIN_PERCENT = 0.1                # min % of diff between foreground and background to detect if motion has stopped
MAX_PERCENT = .0001 #.001 .0001

# params by suraj
MIN_WAIT_BETWEEEN_TWO_SLIDES = 40 #(30 frames per second * 2 seconds)
LOG_TO_FILE_FOR_DEBUG = False
WANT_TO_MANUALLY_SEE_COORDS_OF_SOME_REGION = True

def get_frames(video_path):
    '''A fucntion to return the frames from a video located at video_path
    this function skips frames as defined in FRAME_RATE'''
    
    
    # open a pointer to the video file initialize the width and height of the frame
    vs = cv2.VideoCapture(video_path)
    if not vs.isOpened():
        raise Exception(f'unable to open file {video_path}')


    total_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_time = 0
    frame_count = 0
    print("total_frames: ", total_frames)
    print("FRAME_RATE", FRAME_RATE)

    # loop over the frames of the video
    while True:
        # grab a frame from the video

        vs.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)    # move frame to a timestamp
        frame_time += 1/FRAME_RATE

        (_, frame) = vs.read()
        # if the frame is None, then we have reached the end of the video file
        if frame is None:
            break

        frame_count += 1
        yield frame_count, frame_time, frame

    vs.release()

def only_speaker_mode(frame, frame_count):
    # applies only for shree k nayar's video.
    
    (top_left_x, top_left_y, bottom_right_x, bottom_right_y) = (79,20,470,274)  #(337, 340, 914,378)    
    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    
    # Calculate the Euclidean distance between the ROI pixels and roi[0, 0] in the RGB color space
    distances = np.linalg.norm(roi - roi[0, 0], axis=-1)

    # Define the color similarity threshold
    threshold = 2
    # Check if the distances are below the color similarity threshold
    is_plain = np.all(distances < threshold)
    if is_plain:
        if LOG_TO_FILE_FOR_DEBUG:
            with open("log.txt", "a") as f:
                debug_str="Case0"
                f.write(f"{debug_str}, frame_count={frame_count}, distances_mean={distances.mean()}, is_plain={is_plain}\n")
    return is_plain

def detect_unique_screenshots(video_path, output_folder_screenshot_path):
    ''''''
    # Initialize fgbg a Background object with Parameters
    # history = The number of frames history that effects the background subtractor
    # varThreshold = Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model. This parameter does not affect the background update.
    # detectShadows = If true, the algorithm will detect shadows and mark them. It decreases the speed a bit, so if you do not need this feature, set the parameter to false.

    fgbg = cv2.createBackgroundSubtractorMOG2(history=FGBG_HISTORY, varThreshold=VAR_THRESHOLD,detectShadows=DETECT_SHADOWS)
    captured = False
    start_time = time.time()
    (W, H) = (None, None)

    screenshoots_count = 0
    #hardcode by seeing frames where speaker is present so that speaker movement will be ignored.
    (top_left_x, top_left_y, bottom_right_x, bottom_right_y) = (915, 360, 1279, 719)  #(337, 340, 914,378)
    i=0
    frame_skipped=0
    if LOG_TO_FILE_FOR_DEBUG:
        with open("log.txt", "w") as f: # to overwrite the file
                f.write(f"Log for debugging\n")
    frame_skipped_nayar_video = 0
    for frame_count, frame_time, frame in get_frames(video_path):
        orig = frame.copy() # clone the original frame (so we can save it later), 
        #frame = imutils.resize(frame, width=600) # resize the frame

        if only_speaker_mode(frame,frame_count):
            print("skipping only speaker frames...Applies only for shree k nayar's video.")
            frame_skipped_nayar_video+=1
            frame_skipped+=1
            continue
        if i==0 and WANT_TO_MANUALLY_SEE_COORDS_OF_SOME_REGION: #set i for which you want to see the region corrdinates in a frame
            # Used just to hardcode the speaker region, only speaker frames. 
            # To later ignore them while background subtraction.
            window_name = "Hover Mouse to select the points(top-left,bottom-right) for speaker region"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            window_width = 1366 # see in ubuntu settings
            window_height = 768
            cv2.resizeWindow(window_name, window_width, window_height)
            cv2.imshow(window_name, frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # if i<500:
        #     #save starting 500 frames for debugging purpose.
        #     os.makedirs("temp",exist_ok=True)
        #     path = os.path.join("temp", f"{frame_count:03}_{round(frame_time/60, 4)}.png")
        #     cv2.imwrite(path, orig)

        i+=1
        frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
        mask = fgbg.apply(frame) # apply the background subtractor


        # if the width and height are empty, grab the spatial dimensions
        if W is None or H is None:
            (H, W) = mask.shape[:2]

        # compute the percentage of the mask that is "foreground"
        non_zero_pixels = cv2.countNonZero(mask)
        p_diff = (non_zero_pixels / float(W * H)) * 100

        # if p_diff less than N% then motion has stopped, thus capture the frame
        # if (not captured) and p_diff < MIN_PERCENT  and frame_count > WARMUP:
        if (not captured) and frame_count >= WARMUP and (frame_count==1 or frame_skipped > MIN_WAIT_BETWEEEN_TWO_SLIDES//6):
            #save this frame as a slide. When:
            # If found large movements (once entered case3) and save a frame after skipping few(settled down)
            captured = True
            filename = f"{screenshoots_count:03}_{round(frame_time/60, 2)}_{frame_count:04}.png"
            path = os.path.join(output_folder_screenshot_path, filename)
            print("saving {}".format(path))
            cv2.imwrite(path, orig)
            screenshoots_count += 1
            debug_str = "Case1"
        # otherwise, either the scene is changing or we're still in warmup
        # mode so let's wait until the scene has settled or we're finished
        # building the background model

        elif captured and p_diff >= MAX_PERCENT and frame_skipped > MIN_WAIT_BETWEEEN_TWO_SLIDES:
            # After entering here, Now will save a frame as a slide after few frames(after settling down).
            captured = False
            frame_skipped = 0
            debug_str = "Case2"
        else: 
            # SKIP if: 1. no motion, or 
            # 2. still in warmup mode, or 
            # 3. not settled down yet(showing old movement).
            debug_str = "Case3"
            frame_skipped += 1

        # To debug, what is happening in each frame.
        if LOG_TO_FILE_FOR_DEBUG:
            with open("log.txt", "a") as f:
                f.write(f"{debug_str}, frame_count={frame_count}, non_zero_pixels={non_zero_pixels}, p_diff={p_diff}, frame_skipped={frame_skipped}, captured={captured}, p_diff >= MAX_PERCENT={p_diff >= MAX_PERCENT}, frame_skipped > MIN_WAIT_BETWEEEN_TWO_SLIDES={frame_skipped > MIN_WAIT_BETWEEEN_TWO_SLIDES}\n")

    print(f'{screenshoots_count} screenshots Captured!')
    print(f"frame skipped where only Nayar Sir is present={frame_skipped_nayar_video}, video is 30fps")
    print(f'Time taken {time.time()-start_time}s')
    return 


def initialize_output_folder(video_path):
    '''Clean the output folder if already exists'''
    video_name = ".".join(video_path.rsplit('/')[-1].split('.')[:-1])
    output_folder_screenshot_path = f"{OUTPUT_SLIDES_DIR}/{video_name}"

    if os.path.exists(output_folder_screenshot_path):
        shutil.rmtree(output_folder_screenshot_path)

    os.makedirs(output_folder_screenshot_path, exist_ok=True)
    print('initialized output folder', output_folder_screenshot_path)
    return output_folder_screenshot_path


def convert_screenshots_to_pdf(output_folder_screenshot_path):
    video_name = ".".join(video_path.rsplit('/')[-1].split('.')[:-1])
    output_pdf_path = f"{OUTPUT_SLIDES_DIR}/{video_name}" + '.pdf'
    print('output_folder_screenshot_path', output_folder_screenshot_path)
    print('output_pdf_path', output_pdf_path)
    print('converting images to pdf..')
    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert(sorted(glob.glob(f"{output_folder_screenshot_path}/*.png"))))
    print('Pdf Created!')
    print('pdf saved at', output_pdf_path)


if __name__ == "__main__":
    
#     video_path = "./input/Test Video 2.mp4"
#     choice = 'y'
#     output_folder_screenshot_path = initialize_output_folder(video_path)
    
    
    parser = argparse.ArgumentParser("video_path")
    parser.add_argument("video_path", help="path of video to be converted to pdf slides", type=str)
    args = parser.parse_args()
    video_path = args.video_path

    print('video_path', video_path)
    output_folder_screenshot_path = initialize_output_folder(video_path)
    detect_unique_screenshots(video_path, output_folder_screenshot_path)

    print('Please Manually verify screenshots and delete duplicates')
    while True:
        choice = input("Press y to continue and n to terminate")
        choice = choice.lower().strip()
        if choice in ['y', 'n']:
            break
        else:
            print('please enter a valid choice')

    if choice == 'y':
        convert_screenshots_to_pdf(output_folder_screenshot_path)