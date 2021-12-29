The following program attempts to detect faces and recognize them in real-time if they are present in the database.

Build the program using CMake.
To run the program, specify the path to the executable and (optionally) add name of the video in the "test" directory you wish to process, preceded by the "--video" flag.
If name of the video is not specified, all of the videos will be processed one by one.
The program calculates statistics by default. If you do not have the annotations or just do not want to calculate them, you should add "--skip-stats" flag at the end.

Examples for Windows system (copy and paste to the command line without quotes):
- "build\bin\Release\faceID ";
- "build\bin\Release\faceID --skip-stats";
- "build\bin\Release\faceID --video christian_bale1_fin_s.mp4";
- "build\bin\Release\faceID --video christian_bale1_fin_s.mp4 --skip-stats";

If you wish to close the current video (you will go to the next video automatically if the name of the video was not specified), press the Q button on the keyboard.
If "--skip-stats" flag was not specified, detection and recognition statistics will be printed in the terminal when the video closes.

If you wish to add a new person to the database, create a new folder with the person's name in the "training_set" directory and place the images of that person into the created folder. To extract the descriptors from the images, please follow the instructions from "utilities\trainingsetextractor_instructions.md".

If you wish to add a new testing video, add it to the "test" directory. To annotate it. please follow the instructions from "utilities\videoannotations_instructions.md".

IMPORTANT: OpenCV library must be installed and added to environment variables.

IMPORTANT: As of now, the program recognizes the person's identity by its index, which is extracted from the alphabetic order of the names in the "training_set" folder. Therefore, if you add a new person in the database, you will have to reannotate all the existing videos. I apologize for such inconvinience.

IMPORTANT: If you are building the program on Windows OS, the video will play with audio. On the other operating system, the audio will not be played, but in theory the program should still compile. However, I don't have the ability to test my program on other operating systems, so some errors might occur. If you have them, please delete everything that is preceded by "ifdef _WIN32" directive in "src\VideoProcessing.cpp". This directive occurs 4 times in this file.