from multiprocessing import Process, Event
from loguru import logger


class Processing:

    def __init__(self, event):
        self._running = True
        self.event = event

    def terminate(self):
        self.event.set()

    def analyse_trend(self, werte):
        if len(werte) >= 4:
            # Die letzten vier Werte extrahieren
            letzten_vier_werte = werte[-4:]

            # Überprüfen, ob die Werte steigen oder fallen
            steigen = all(x < y for x, y in zip(letzten_vier_werte, letzten_vier_werte[1:]))
            fallen = all(x > y for x, y in zip(letzten_vier_werte, letzten_vier_werte[1:]))

            # Überprüfen, ob die letzten Werte größer oder kleiner 0 sind
            groesser_null = True if letzten_vier_werte[-1] > 0 else False
            kleiner_null = True if letzten_vier_werte[-1] < 0 else False

            # Entsprechenden Wert zurückgeben (0: Holding; 1: Shifting)
            if groesser_null and steigen:
                return 1
            elif kleiner_null:
                return 0
            elif groesser_null and fallen:
                return 0

        # Default-Wert, falls nicht genügend Werte vorhanden sind
        return 0
    
    def analyse_headpose(self, headpose_list):
        # Prüfe die letzten 10 Werte in der Liste
        letzten_10_werte = headpose_list[-10:]
        
        # Überprüfe, ob sich der Wert von 0 auf 1 geändert hat
        if 0 in letzten_10_werte and 1 in letzten_10_werte and letzten_10_werte.index(0) > letzten_10_werte.index(1):
            return 2
        elif headpose_list[-1] == 1:
            return 1
        else:
            return 0

    def run(self, ListOfQueues, OutputQueue):
        logger.debug("Process enters loop")
        diff_list = []
        headpose_list = []
        while not self.event.is_set():
            VAP_out = ListOfQueues[0].get().tolist()
            #logger.debug("VAP: {}", VAP_out)
            # compute the difference - if diff > 0: hold else: shift
            diff = VAP_out[1] - VAP_out[0]
            diff_list.append(diff)
            # Analyse durchführen
            VAP = self.analyse_trend(diff_list)
            #logger.debug("VAP: {}", VAP)

            Prosodie_out = ListOfQueues[1].get()
            # logger.debug("Prosodie: {}", Prosodie_out)

            Headpose_out = ListOfQueues[2].get()
            headpose_list.append(Headpose_out)
            Headpose = self.analyse_headpose(headpose_list)
            logger.debug("Headpose: {}", Headpose)

            if (VAP + Prosodie_out + Headpose) >= 3:
                ergebnis = 1
            else:
                ergebnis = 0
            
            OutputQueue.put(Headpose)


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
