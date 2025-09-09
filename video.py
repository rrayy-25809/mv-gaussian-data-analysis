import cv2
import main

def video_to_frames(video_filename):
    vidcap = cv2.VideoCapture(video_filename)
    success, image = vidcap.read()
    images = []
    while success:
        images.append(image)
        success, image = vidcap.read()
    return images

def transform_images(images):
    return [cv2.cvtColor(main.사진반환(image), cv2.COLOR_BGR2RGB) for image in images]

def frames_to_video(images, output_filename):
    height, width, layers = images[0].shape
    size = (width,height)
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    for i in range(len(images)):
        out.write(images[i])
    out.release()

#사진변환
images = video_to_frames('ex.mp4')

#사진변형
transformed_images = transform_images(images)

#사진조립
frames_to_video(transformed_images, 'output.mp4')
