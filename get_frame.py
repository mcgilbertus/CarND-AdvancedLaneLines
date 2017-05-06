import sys
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("usage: get_frame movie time [frame]")
    sys.exit(1)
movie_file = sys.argv[1]
frame_time = sys.argv[2]
if len(sys.argv) > 3:
    frame_name = sys.argv[3]
else:
    frame_name = "frame"+frame_time.replace(":","_")+".jpg"

video = VideoFileClip(movie_file)
frame = video.get_frame(frame_time)
plt.imshow(frame)
plt.show()
video.save_frame(frame_name, t=frame_time)
print("Frame saved as %s" %frame_name)
