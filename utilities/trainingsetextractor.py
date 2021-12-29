# The following script is used to extract SIFT descriptors from the photo.
# Specify the path to the person's images when starting the script from the command line.
#
# Examples:
# 'python utilities\trainingsetextractor.py "training_set\Christian Bale"';
# 'python utilities\trainingsetextractor.py "training_set\Harrison Ford"'.
#
# Select the face with mouse. Only one face may be selected on an image. Press SPACE, ESCAPE or ENTER key to continue.
# To skip the image, do not select anything, and press SPACE, ESCAPE or ENTER key.
# The descriptors are saved in the same directory in "descriptors\SIFT" subfolder.
#
# OpenCV library must be installed. To install it, run the "pip install opencv-python" command in the terminal.

import cv2
from glob import glob
from os.path import isdir
from os import makedirs
from sys import argv

class IncorrectNumberOfArguments(Exception):
    pass

class NoDirectory(Exception):
    pass

class UnacceptableFeatureExtractorType(Exception):
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
    return (x_top, y_top), (x_bottom, y_bottom)

def descriptor_filename(image_filename):
    """
    Returns path where image will be saved.

    """
    path = image_filename.split("\\")
    if not isdir(f"{directory}\\descriptors\\SIFT"):
        makedirs(f"{directory}\\descriptors\\SIFT")
    path[-1] =  f"descriptors\\SIFT\\{path[-1][:-4]}_descriptor_SIFT.png"        
    return "\\".join(path)

def save_descriptor(image_name, face):
    """
    Saves the descriptor to ".png" image for easy loading in the main program.
    
    """
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(face, None)            
    cv2.imwrite(descriptor_filename(image_name), descriptors)


if __name__ == "__main__":
    
    if len(argv) != 2:
        raise IncorrectNumberOfArguments('Only the path to images must be specified.')
    
    directory = argv[1]
    if not isdir(directory):
        raise NoDirectory('No such directory exists.')
    
    is_drawing = False
    x_initial = -1
    y_initial = -1
    
    # Get images present in the directory:
    image_names = glob(f'{directory}\\*.jpg') + glob(f'{directory}\\*.png')
    
    for image_name in image_names:
        image = cv2.imread(image_name)
        
        # Downscale the image in case it cannot fit the screen:
        scale = 1
        if image.shape[1] > 1500 or image.shape[0] > 800:
            scale_x = 1500 / image.shape[1]
            scale_y = 800 / image.shape[0]
            scale = min(scale_x, scale_y)
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
            image = cv2.resize(image, (width, height))
        cache = image.copy()
        
        top_corner = -1
        bottom_corner = -1
        
        already_drawn = False

        def draw_square(event, x, y, flags, param):
            """
            Function that actually draws square.
            
            """
            global is_drawing, x_initial, y_initial
            global image, cache, square_parameters
            global top_corner, bottom_corner
            global already_drawn            
                
            if event == cv2.EVENT_LBUTTONDOWN:
                if already_drawn:
                    cv2.putText(image, 'There must be only one face', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    is_drawing = True
                    x_initial = x
                    y_initial = y
                
            elif event == cv2.EVENT_LBUTTONUP:
                if already_drawn:
                    cv2.putText(image, 'There must be only one face', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    is_drawing = False
                    top_corner, bottom_corner = square_params(x_initial, x, y_initial, y)
                    cv2.rectangle(image, top_corner, bottom_corner, color=(0, 0, 255), thickness=2)
                    already_drawn = True
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if is_drawing:
                    image = cache.copy()
                    top_corner, bottom_corner = square_params(x_initial, x, y_initial, y)
                    cv2.rectangle(image, top_corner, bottom_corner, color=(0, 0, 255), thickness=2)
                     
        cv2.namedWindow(image_name)
        cv2.setMouseCallback(image_name, draw_square)

        while True:
            cv2.imshow(image_name, image)
            if cv2.waitKey(5) & 0xFF in (13, 32, 27):
                break
        
        if already_drawn:
            face = cache[top_corner[1]:bottom_corner[1], top_corner[0]:bottom_corner[0], :]        
            save_descriptor(image_name, face)
        
        cv2.destroyAllWindows()