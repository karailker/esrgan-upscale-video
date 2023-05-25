import cv2
import moviepy.video.io.ImageSequenceClip as ImageSequenceClip
import moviepy.video.io.VideoFileClip as VideoFileClip
import moviepy.audio.io.AudioFileClip as AudioFileClip
import moviepy.audio.AudioClip as AudioClip

import os
import glob


import tempfile
from tqdm import tqdm

def frames_to_video(out_folder, filename, fps, audio_file):
    image_files = []

    for file in glob.glob(f'{out_folder}/*.png'):
        image_files.append(file)

    image_files.sort(key = lambda x: int(x.split('/')[-1].split('.')[0]))

    clip = ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    
    audio_clip = AudioClip.CompositeAudioClip([AudioFileClip.AudioFileClip(audio_file),])
    
    clip.audio = audio_clip
    
    clip.write_videofile(filename)

def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where 
    # x is the frame index
    
    vidcap = cv2.VideoCapture(video)
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total = total_frame_count, desc="Extracting frames") as pbar:
        count = 0
        while vidcap.isOpened():
            success, image = vidcap.read()
            if success:
                cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
                count += 1
                # print(f'{count}.png')
                pbar.update(count)
            else:
                break
    cv2.destroyAllWindows()
    vidcap.release()
    
def video_to_audio(video):
    clip = VideoFileClip.VideoFileClip(video)
    # temp = tempfile.NamedTemporaryFile(suffix=".mp3")
    temp_file_name = None
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmpfile:
        temp_file_name = tmpfile.name
    clip.audio.write_audiofile(temp_file_name)
    
    return temp_file_name

def video_framerate(video):
    vidcap = cv2.VideoCapture(video)

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = vidcap.get(cv2.CAP_PROP_FPS)

    vidcap.release()

    return fps
