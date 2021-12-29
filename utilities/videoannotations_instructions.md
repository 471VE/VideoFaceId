IMPORTANT: as of now, this script does NOT support more than one unknown person on each frame. Please keep that in mind.

The following script is used to annotate the real face positions on the video.
Specify the name of the video in the "test" directory in the command line. In case it isn't specified, all videos in the "test" folder will be opened one by one.

Examples:
'python utilities\videoannotations.py';
'python utilities\videoannotations.py christian_bale1_fin_s.mp4'.

At first, you have to specify the indexes of available persons that are present in the video selected, separated by the space symbol. Then, the video will be opened exactly the number of selected persons times. You will have to select only one person's face on each play.

Select the face with mouse. Only one face may be selected on the frame. Press SPACE or L key to continue. After that, tracker will follow the face until it fails to do so. You will have to select the face again or skip the frame until the one where you would want to select the face.

To skip the frame, do not select anything, and press SPACE or L key. Press K key to skip 5 frames. Press J key to go to the previous frame.
The annotations are saved in the same directory in "annotations" subfolder in the folder that matches the name of the video.

It shouldn't take more than one minute to annotate one video of less than 20 seconds.

OpenCV library must be installed. To install it, run the "pip install opencv-python" command in the terminal.