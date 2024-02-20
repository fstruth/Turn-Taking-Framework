import pyaudio
from multiprocessing import Process, Event
from loguru import logger


class AUDIO_READ:

    def __init__(self, event):
        self._running = True
        self.event = event

    def terminate(self):
        self.event.set()

    def run(self, CHUNK_OutputQueue, audiodevice):
        # Do Setup stuff
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100

        p = pyaudio.PyAudio()
        # find the right device!
        count = p.get_device_count()
        device = []
        DeviceIndex = 0
        for i in range(count):
            device = p.get_device_info_by_index(i)
            logger.debug(str(device["name"]))
            if device["name"] == audiodevice:
                DeviceIndex = i
                break

        # print(p.get_default_input_device_info())
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        input_device_index=DeviceIndex,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        logger.debug("* recording")
        while not self.event.is_set():
            # Capture the audio
            data = stream.read(CHUNK, exception_on_overflow=False)
            # logger.debug("recorded chunk")
            CHUNK_OutputQueue.put(data)

        # After the loop release the used objects
        logger.debug("* done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()


event = Event()
c = AUDIO_READ(event)
t = None


def StopAUDIO_READ():
    global c, t
    # Signal termination
    c.terminate()
    logger.debug("Process terminated")


def StartAUDIO_READ(audiodevice="", OutputQueue=None):
    global t, event, c
    # c = VideoRecorder_WebCam()
    if t is None:
        event = Event()
        c = AUDIO_READ(event)
        t = Process(target=c.run, args=(OutputQueue, audiodevice))
        t.start()
        logger.debug("Process started")
