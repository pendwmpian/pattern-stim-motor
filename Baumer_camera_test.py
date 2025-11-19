import neoapi
import sys
import time
import cv2
import os
import datetime
import skvideo.io

camera = neoapi.Cam()
camera.Connect()
camera.SetImageBufferCount(3000)       # set the size of the buffer queue to 50
camera.SetImageBufferCycleCount(3000)  # and the cycle count as well

CAMERA_WIDTH = 480
CAMERA_HEIGHT = 480

img = []

if camera.IsConnected():
    camera.f.TriggerMode.value = neoapi.TriggerMode_Off
    camera.f.AcquisitionFrameRateEnable = True
    camera.f.AcquisitionFrameRate = 1000

    camera.f.ExposureTime.Set(900)
    camera.f.Gain.Set(5.5)
    camera.f.Width.Set(CAMERA_WIDTH)
    camera.f.Height.Set(CAMERA_HEIGHT)
    camera.f.OffsetX.Set(960 - CAMERA_WIDTH // 2)
    camera.f.OffsetY.Set(540 - CAMERA_HEIGHT // 2)

    print(time.time())

    for i in range(3000):
        try:
            img.append(camera.GetImage()) 
        except (neoapi.NoImageBufferException) as exc:
            print(sys.exc_info()[0])
            print("NoImageBufferException: ", exc)

    print(time.time())

path = "./img/" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(path, exist_ok=True)
# for i in range(3000):
#     cv2.imwrite(path + "/{:0=4}.png".format(i), img[i].GetNPArray())	

# for video compression

output_video = path + "/3000.mp4"
output_parameters = {
    '-r': '1000',          # Output FPS
    '-c:v': 'libx264',     # H.264 codec
    '-crf': '17',           # Lossless mode
    '-preset': 'veryfast', # Best compression
    '-pix_fmt': 'gray'  # Standard pixel format (compatible with most players)
}

with skvideo.io.FFmpegWriter(output_video, outputdict=output_parameters) as writer:
    for i in range(3000):
        writer.writeFrame(img[i].GetNPArray())	
