# Visual reconstruction of played music

### Project
Analyze a video of a person playing a keyboard-based musical instrument (such as a piano or accordion) to identify the pressed keys by visually examining both the movements of the fingers and the layout of the keyboard.


The example of the output of the algorithm.

![classify.gif](gifs/classify.gif)

### Team
- Alisa Pesotskaia
- Refika Kalyoncu

### Directory overview

* [/ref](https://github.com/allythe/PianoPressedKeysDetection/tree/main/ref) - reference articles
* [/src](https://github.com/allythe/PianoPressedKeysDetection/tree/main/src) - source code
* [/gifs](https://github.com/allythe/PianoPressedKeysDetection/tree/main/videos) - best result visualization

### Main code

* [main.py](https://github.com/allythe/PianoPressedKeysDetection/blob/main/main.py) - find pressed keys on video

### What to write in the ***params*** in ***main.py***
* video_path - path to the video with played music ( videos/... )
* frame_per_second - number of frames per second ( from 1 to 25 )
* max_number_frames - maximum number of extracted frames ( from 1 )
* keys_extraction_type - type of keys extractor from the image without hands
  * lines - this method uses conventional CV approach 
* hands_extraction_type - type of hands extractor 
  * same - needs to fingers_extraction_type = mediapipe, does not change the input image
  * opencv - utilises conventional CV approach (thresholding)
* fingers_extraction_type
  * opencv - uses convex hull to find fingers from hands masks
  * mediapipe - uses mediapipe library 
* pressed_key_extraction_type
  * mediapipez - uses estimated be mediapipe z-coordinate of fingertips (does not work good)
  * mediapipejoints - uses distances between first and second set of dots on fingers, extracted via mediapipe. Also, uses shade extraction to identify pressed keys by 1 and 5 fingers
  * classify - uses trained classification network to classify image of the probably pressed key
* plot_fingertips - True if plot extracted fingertips
* plot_keys - True if plot extracted keys 


Link to the [Google docs](https://docs.google.com/document/d/1feF4ccT7vdFtKw7OuJex0GS3QpIQq3dwpXOVX01vnfY/edit?tab=t.0) with the raw discussion of our ideas
