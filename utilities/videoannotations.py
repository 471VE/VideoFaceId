# IMPORTANT: as of now, this script does NOT support more than one unknown person on each frame. Please keep that in mind.
#
# The following script is used to annotate the real face positions on the video.
# Specify the name of the video in the "test" directory in the command line.
# In case it isn't specified, all videos in the "test" folder will be opened one by one.
#
# Examples:
# 'python utilities\videoannotations.py';
# 'python utilities\videoannotations.py christian_bale1_fin_s.mp4'.
#
# At first, you have to specify the indexes of available persons that are present in the video selected, separated by the space symbol.
# Then, the video will be opened exactly the number of selected persons times.
# You will have to select only one person's face on each play.
#
# Select the face with mouse. Only one face may be selected on the frame. Press SPACE or L key to continue.
# After that, tracker will follow the face until it fails to do so.
# You will have to select the face again or skip the frame until the one where you would want to select the face.
# To skip the frame, do not select anything, and press SPACE or L key.
# Press K key to skip 5 frames. Press J key to go to the previous frame.
# The annotations are saved in the same directory in "annotations" subfolder in the folder that matches the name of the video.
#
# It shouldn't take more than one minute to annotate one video of less than 20 seconds.
#
# OpenCV library must be installed. To install it, run the "pip install opencv-python" command in the terminal.

import cv2
from glob import glob
from os.path import isdir
from os import makedirs
from sys import argv


class IncorrectNumberOfArguments(Exception):
    pass


def square_params(x_initial, x, y_initial, y):
    """
    Calculates square parameters acquired from the mouse movements for rendering the square on the image.
    
    """
    side = abs(y_initial - y)    
    x_top = round(x - side/2)
    x_bottom = round(x + side/2)    
    y_top = min(y_initial, y)
    y_bottom = max(y_initial, y)    
    return (x_top, y_top), (x_bottom, y_bottom), side


def annotations_filename(video_name, person_name):
    """
    Returns the filename where the annotations will be saved.
    
    """
    path = video_name.split("\\")
    if not isdir(f"test\\annotations\\{path[-1][:-4]}"):
        makedirs(f"test\\annotations\\{path[-1][:-4]}")
    filename =  f"test\\annotations\\{path[-1][:-4]}\\annotation_{person_name}.txt"
    return filename


if __name__ == "__main__":
    
    if len(argv) not in (1, 2):
        raise IncorrectNumberOfArguments('Only the path to video may be specified.')
    
    is_drawing = False
    x_initial = -1
    y_initial = -1
    
    if len(argv) == 1:
        video_names = glob('test\\*.mp4')
    elif len(argv) == 2:
        video_names = [f"test\\{argv[1]}"]
    
    # Load names of persons available in the dataset:
    instruction_names = glob('training_set\\*')
    instruction_names = list(map(lambda x: x.split('\\')[-1], instruction_names)) + ['Unknown']
    instruction_list = '\n'.join([f"{name}: {index}" for index, name in enumerate(instruction_names)])    

    for video_name in video_names:
        instructions = f"\nPlease specify the names of people present in the \"{video_name}\""
        instructions += ", separated with space, according to the following table:\n"
        instructions += f"{instruction_list}\n"
        
        names = input(instructions)
        names = names.split()
        exit_video = False
        write_to_file = True
        
        for person_name in names:
            print(instruction_names[int(person_name)])
            tracker = cv2.TrackerKCF_create()
            tracked = False

            capture = cv2.VideoCapture(video_name)
            annotations_string = f"{person_name}\n"
            
            face_box = 0
            
            while capture.isOpened():
                if exit_video:
                    exit_video = True
                    write_to_file = False
                    break
                
                ret, frame = capture.read()
                if not ret:
                    print("Video end. Exiting ...")
                    break
                
                if tracked:
                    ok, face_box = tracker.update(frame)
                    
                    if not ok:                        
                        cache = frame.copy()
                        top_corner = -1
                        bottom_corner = -1
                        side = -1
                        already_drawn = False

                        def draw_square(event, x, y, flags, param):
                            """
                            Function that draws the square on the frame.
                            
                            """
                            global is_drawing, x_initial, y_initial
                            global frame, cache
                            global top_corner, bottom_corner, side   
                            global already_drawn       
                                
                            if event == cv2.EVENT_LBUTTONDOWN:
                                if already_drawn:
                                    cv2.putText(frame, 'There must be only one face', (50, 50),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                else:
                                    is_drawing = True
                                    x_initial = x
                                    y_initial = y
                                
                            elif event == cv2.EVENT_LBUTTONUP:
                                if already_drawn:
                                    cv2.putText(frame, 'Only one face must be selected', (50, 50),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                else:
                                    is_drawing = False
                                    top_corner, bottom_corner, side = square_params(x_initial, x, y_initial, y)
                                    cv2.rectangle(frame, top_corner, bottom_corner, color=(0, 0, 255), thickness=2)
                                    already_drawn = True
                                
                            elif event == cv2.EVENT_MOUSEMOVE:
                                if is_drawing:
                                    frame = cache.copy()
                                    top_corner, bottom_corner, side = square_params(x_initial, x, y_initial, y)
                                    cv2.rectangle(frame, top_corner, bottom_corner, color=(0, 0, 255), thickness=2)
                                    
                        cv2.namedWindow(video_name)
                        cv2.setMouseCallback(video_name, draw_square)

                        while True:
                            cv2.imshow(video_name, frame)
                            keyboard = cv2.waitKey(1) & 0xFF
                            
                            if keyboard in (32, ord('l')): # SPACE key or L key
                                break
                            
                            elif keyboard == 27: # ESCAPE key
                                exit_video = True
                                break
                            
                            elif keyboard == ord('j'): # J key
                                next_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
                                previous_frame = next_frame - 2
                                capture.set(cv2.CAP_PROP_POS_FRAMES, previous_frame)
                                prev_frame = True
                                break
                            
                            elif keyboard == ord('k'): # K key
                                next_frame = capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
                                capture.set(cv2.CAP_PROP_POS_FRAMES, next_frame + 5)
                                skip_frames = True
                                tracked = False
                                break

                        if already_drawn:
                            annotations_string += f"{top_corner[0]} {top_corner[1]} {side}\n"
                            face_box = (top_corner[0], top_corner[1], side, side)
                            tracker = cv2.TrackerKCF_create()
                            tracker.init(frame, face_box)
                            tracked = True
                        else:
                            if prev_frame:
                                prev_frame = False
                                annotations_string = annotations_string[:-1]
                            elif skip_frames:
                                skip_frames = False
                                annotations_string += "\n\n\n\n\n"
                            else:
                                annotations_string += "\n"
                                tracked = False
                            
                    elif ok:
                        annotations_string += f"{face_box[0]} {face_box[1]} {face_box[2]}\n"
                        top_corner = (int(face_box[0]), int(face_box[1]))
                        bottom_corner = (int(face_box[0] + face_box[2]), int(face_box[1] + face_box[3]))
                        cv2.rectangle(frame, top_corner, bottom_corner, color=(0, 0, 255), thickness=2)
                        cv2.imshow(video_name, frame)
                        cv2.waitKey(1)

                elif not tracked:
                    # Giant chunck of code that had to be repeated beacause of difficulties with dealing with global variables:
                    cache = frame.copy()
                    top_corner = -1
                    bottom_corner = -1
                    side = -1
                    already_drawn = False

                    def draw_square(event, x, y, flags, param):
                        global is_drawing, x_initial, y_initial
                        global frame, cache
                        global top_corner, bottom_corner, side   
                        global already_drawn       
                            
                        if event == cv2.EVENT_LBUTTONDOWN:
                            if already_drawn:
                                cv2.putText(frame, 'There must be only one face', (50, 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            else:
                                is_drawing = True
                                x_initial = x
                                y_initial = y
                            
                        elif event == cv2.EVENT_LBUTTONUP:
                            if already_drawn:
                                cv2.putText(frame, 'Only one face must be selected', (50, 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            else:
                                is_drawing = False
                                top_corner, bottom_corner, side = square_params(x_initial, x, y_initial, y)
                                cv2.rectangle(frame, top_corner, bottom_corner, color=(0, 0, 255), thickness=2)
                                already_drawn = True
                            
                        elif event == cv2.EVENT_MOUSEMOVE:
                            if is_drawing:
                                frame = cache.copy()
                                top_corner, bottom_corner, side = square_params(x_initial, x, y_initial, y)
                                cv2.rectangle(frame, top_corner, bottom_corner, color=(0, 0, 255), thickness=2)
                                
                    cv2.namedWindow(video_name)
                    cv2.setMouseCallback(video_name, draw_square)

                    prev_frame = False
                    skip_frames = False
                    while True:
                        cv2.imshow(video_name, frame)
                        keyboard = cv2.waitKey(1) & 0xFF
                        
                        if keyboard in (32, ord('l')): # SPACE key or L key
                            break
                        
                        elif keyboard == 27: # ESCAPE key
                            exit_video = True
                            break
                        
                        elif keyboard == ord('j'): # J key
                            next_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
                            previous_frame = next_frame - 2
                            capture.set(cv2.CAP_PROP_POS_FRAMES, previous_frame)
                            prev_frame = True
                            break
                        
                        elif keyboard == ord('k'): # K key
                            next_frame = capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
                            capture.set(cv2.CAP_PROP_POS_FRAMES, next_frame + 5)
                            skip_frames = True
                            tracked = False
                            break

                    if already_drawn:
                        annotations_string += f"{top_corner[0]} {top_corner[1]} {side}\n"
                        face_box = (top_corner[0], top_corner[1], side, side)
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, face_box)
                        tracked = True
                    else:
                        if prev_frame:
                            prev_frame = False
                            annotations_string = annotations_string[:-1]
                        elif skip_frames:
                            skip_frames = False
                            annotations_string += "\n\n\n\n\n"
                        else:
                            annotations_string += "\n"
                            tracked = False
                
            cv2.destroyAllWindows()
            if write_to_file:
                # Save annotations to file:           
                with open(annotations_filename(video_name, person_name), "w") as annotations_file:
                    annotations_file.write(annotations_string[:-1])
            else:
                write_to_file = True