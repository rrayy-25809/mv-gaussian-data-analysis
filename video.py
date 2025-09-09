import cv2
from main import 사진반환 # 쓸데없이 함수 모듈을 전부 불러오지 않고 함수만 가져오기

def video_to_frames(video_filename):
    vidcap = cv2.VideoCapture(video_filename)
    success, image = vidcap.read()
    images = []
    while success: # 이거 아이디어 좋다
        images.append(image)
        success, image = vidcap.read()
    return images

def transform_images(images):
    return [cv2.cvtColor(사진반환(image), cv2.COLOR_BGR2RGB) for image in images]

def frames_to_video(images, output_filename):
    height, width, layers = images[0].shape
    size = (width,height)
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter.fourcc(*'mp4v'), 30, size) # 4.12 버전에서는 VideoWriter 객체 안으로 함수가 이동됨
    for i in images:
        out.write(i)
    out.release()

if __name__ == "__main__":
    #사진반환
    images = video_to_frames('video.mp4')

    #사진변형
    transformed_images = transform_images(images)

    #사진조립
    frames_to_video(transformed_images, 'output.mp4')
