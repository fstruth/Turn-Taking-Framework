from multiprocessing import Process, Event
from loguru import logger

strModNr = 20


class AudioChunkBroker:

    def __init__(self, event):
        self._running = True
        self.event = event
        self.Cleared = False

    def terminate(self):
        self.event.set()
        # = False

    def run(self, AudioBrokerInputQueue, ListOfQueues):
        logger.debug("Process enters loop")
        while not self.event.is_set():
            try:
                if not AudioBrokerInputQueue.empty():
                    # distribute to all Threads
                    # run Person detection
                    Chunk = AudioBrokerInputQueue.get()
                    # Verteile das Chunk
                    for queue in ListOfQueues:
                        queue.put(Chunk)
            except Exception as e:
                # Hier wird der Code ausgeführt, um die Exception abzufangen
                logger.debug("Fehler aufgetreten:  " + str(e))
                break


event = Event()
c = AudioChunkBroker(event)
t = None


def StopAudioChunkBroker():
    global c, t
    # Signal termination
    c.terminate()
    logger.debug("Process terminated")


def StartAudioChunkBroker(AudioBrokerInputQueue, ListOfQueues):
    global t, event, c
    if t is None:
        event = Event()
        c = AudioChunkBroker(event)
        t = Process(target=c.run, args=(AudioBrokerInputQueue, ListOfQueues))
        t.start()
        logger.debug("Process started")
