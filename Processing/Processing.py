from multiprocessing import Process, Event
from loguru import logger


class Processing:

    def __init__(self, event):
        self._running = True
        self.event = event
        self.diff_list = []

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
            groesser_null = all(x > 0 for x in letzten_vier_werte)
            kleiner_null = all(x < 0 for x in letzten_vier_werte)

            # Entsprechenden Wert zurückgeben (0: Holding; 1: Shifting)
            if groesser_null and steigen:
                return 1
            elif kleiner_null:
                return 0
            elif groesser_null and fallen:
                return 0

        # Default-Wert, falls nicht genügend Werte vorhanden sind
        return None

    def run(self, ListOfQueues, OutputQueue):
        logger.debug("Process enters loop")
        while not self.event.is_set():
            VAP_out = ListOfQueues[0].get()
            # compute the difference - if diff > 0: hold else: shift
            diff = VAP_out[1] - VAP_out[0]
            self.diff_list = self.diff_list.append(diff)
            # Analyse durchführen
            VAP = self.analyse_trend(self.diff_list)

            Prosodie_out = ListOfQueues[1].get()

            if sum(VAP, Prosodie_out) == 2:
                ergebnis = 1
            else:
                ergebnis = 0

            OutputQueue.put(ergebnis)


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
