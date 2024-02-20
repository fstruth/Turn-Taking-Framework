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
            # Wenn die letzten beiden Werte kleiner 0.5 sind, findet ein turn-shift statt
            if VAP_list[-1] < 0.5 and VAP_list[-2] < 0.5:
                return 1
            else:
                return 0

        # Default-Wert, falls nicht genügend Werte vorhanden sind
        return 0

    def analyse_headpose(self, headpose_list):

        def finde_index_der_nullen(lst):
            if 0 in lst:
                index_der_letzten_null = len(lst) - 1 - lst[::-1].index(0)
                index_erste_null = lst.index(0) if 0 in lst else None
                return index_erste_null, index_der_letzten_null
            else:
                return None, None

        if len(headpose_list) >= 5:
            # Prüfe die letzten 5 Werte in der Liste
            letzten_5_werte = headpose_list[-5:]

            index_erste_null, index_der_letzten_null = finde_index_der_nullen(letzten_5_werte)

            # Überprüfe, ob sich der Wert von 0 auf 1 geändert hat
            if index_der_letzten_null is not None and all(x == 1 for x in letzten_5_werte[index_der_letzten_null+1:]) and index_erste_null is not None and all(x == 1 for x in letzten_5_werte[index_erste_null+1:]):
                time.sleep(0.1)
                return 2
            elif headpose_list[-1] == 1 and headpose_list[-2] == 1:
                return 1
            else:
                return 0
        return 0

    def run(self, ListOfQueues, OutputQueue):
        logger.debug("Process enters loop")
        VAP_list = []
        headpose_list = []
        while not self.event.is_set():
            VAP_out = ListOfQueues[0].get().tolist()

            VAP_list.append(VAP_out[0])
            # Einzelentscheidung VAP treffen auf Basis der letzten X Werte
            VAP = self.analyse_trend(VAP_list)
            # logger.debug("VAP: {}", VAP)

            # Einzelentscheidung Prosodie ziehen
            Prosodie_out = ListOfQueues[1].get()
            # logger.debug("Prosodie_out: {}", Prosodie_out)

            Headpose_out = ListOfQueues[2].get()
            headpose_list.append(Headpose_out)
            # Einzelentscheidung Headpose treffen auf Basis der letzten X Kopfpositionen
            Headpose = self.analyse_headpose(headpose_list)
            # logger.debug("Headpose: {}", Headpose)

            # Next part makes the final decision - change here the weights or the threshold for turn-shift (1)
            if (1*VAP + 1*Headpose + 1*Prosodie_out) >= 3:
                ergebnis = 1
            else:
                ergebnis = 0

            OutputQueue.put(ergebnis)
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
