The following script is used to extract SIFT descriptors from the photo.
Specify the path to the person's images when starting the script from the command line.

Examples:
'python utilities\trainingsetextractor.py "training_set\Christian Bale"';
'python utilities\trainingsetextractor.py "training_set\Harrison Ford"'.

Select the face with mouse. Only one face may be selected on an image. Press SPACE, ESCAPE or ENTER key to continue.
To skip the image, do not select anything, and press SPACE, ESCAPE or ENTER key.
The descriptors are saved in the same directory in "descriptors\SIFT" subfolder.

OpenCV library must be installed. To install it, run the "pip install opencv-python" command in the terminal.