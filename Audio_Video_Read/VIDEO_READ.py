import cv2
from multiprocessing import Process, Event
from loguru import logger


class VIDEO_READ:

    def __init__(self, event):
        self._running = True
        self.event = event

    def terminate(self):
        self.event.set()

    def run(self, OutputQueue, videodevice):
        # define a video capture object
        vid = cv2.VideoCapture(0)
        vid.set(cv2.CAP_PROP_FPS, 2)

        while not self.event.is_set():
            # Capture the video frame by frame
            ret, frame = vid.read()

            OutputQueue.put(frame)

        # After the loop release the cap object
        vid.release()


event = Event()
c = VIDEO_READ(event)
t = None


def StopVIDEO_READ():
    global c, t
    # Signal termination
    c.terminate()
    logger.debug("Process terminated")


def StartVIDEO_READ(videodevice="", OutputQueue=None):
    global t, event, c
    # c = VideoRecorder_WebCam()
    if t is None:
        event = Event()
        c = VIDEO_READ(event)
        t = Process(target=c.run, args=(OutputQueue, videodevice))
        t.start()
        logger.debug("Process started")
