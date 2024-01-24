from multiprocessing import Process, Event
from loguru import logger
import time


class Processing:

    def __init__(self, event):
        self._running = True
        self.event = event

    def terminate(self):
        self.event.set()

    def analyse_trend(self, VAP_list):
        if len(VAP_list) >= 2:
            if VAP_list[-1] < 0.5 and VAP_list[-2] < 0.5:
                return 1
            else:
                return 0

        # Default-Wert, falls nicht genügend Werte vorhanden sind
        return 0
    
    def analyse_headpose(self, headpose_list):
        # Prüfe die letzten 5 Werte in der Liste
        letzten_5_werte = headpose_list[-5:]
        
        # Überprüfe, ob sich der Wert von 0 auf 1 geändert hat
        if 0 in letzten_5_werte and 1 in letzten_5_werte and letzten_5_werte.index(0) > letzten_5_werte.index(1):
            return 2
        elif headpose_list[-1] == 1:
            return 1
        else:
            return 0

    def run(self, ListOfQueues, OutputQueue):
        logger.debug("Process enters loop")
        VAP_list = []
        headpose_list = []
        while not self.event.is_set():
            VAP_out = ListOfQueues[0].get().tolist()
            # Choose first speaker - if VAP_out[0] > 0.5: hold else VAP_out[0] < 0.5: shift
            VAP_list.append(VAP_out[0])
            # Analyse durchführen
            VAP = self.analyse_trend(VAP_list)
            # logger.debug("VAP: {}", VAP)

            Prosodie_out = ListOfQueues[1].get()
            logger.debug("Prosodie: {}", Prosodie_out)

            Headpose_out = ListOfQueues[2].get()
            headpose_list.append(Headpose_out)
            Headpose = self.analyse_headpose(headpose_list)
            # logger.debug("Headpose: {}", Headpose)

            if (VAP + Prosodie_out + Headpose) >= 3:
                ergebnis = 1
            else:
                ergebnis = 0
            
            OutputQueue.put(Prosodie_out)
            time.sleep(0.1)


event = Event()
c = Processing(event)
t = None


def StopProcessing():
    global c, t
    # Signal termination
    c.terminate()
    logger.debug("Process terminated")


def StartProcessing(ListOfQueues, OutputQueue=None):
    global t, event, c
    if t is None:
        event = Event()
        c = Processing(event)
        t = Process(target=c.run, args=(ListOfQueues, OutputQueue))
        t.start()
        logger.debug("Process started")
