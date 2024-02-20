from multiprocessing import Process, Event
from loguru import logger
import zmq


class Sending:

    def __init__(self, event):
        self._running = True
        self.event = event
        self.diff_list = []

    def terminate(self):
        self.event.set()

    def get_zmq_tcp_connection(self, ip_outsocket):
        # Output results using a PUB socket
        context = zmq.Context()
        outsocket = context.socket(zmq.PUB)
        outsocket.bind("tcp://" + ip_outsocket + ":" + '9999')
        return outsocket

    def run(self, InputQueue, OutputQueue):
        logger.debug("Process enters loop")

        outsocket = self.get_zmq_tcp_connection("127.0.0.1")

        while not self.event.is_set():
            trp = str(InputQueue.get())
            outsocket.send_string(trp)


event = Event()
c = Sending(event)
t = None


def StopSending():
    global c, t
    # Signal termination
    c.terminate()
    logger.debug("Process terminated")


def StartSending(InputQueue=None, OutputQueue=None):
    global t, event, c
    if t is None:
        event = Event()
        c = Sending(event)
        t = Process(target=c.run, args=(InputQueue, OutputQueue))
        t.start()
        logger.debug("Process started")
