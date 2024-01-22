from multiprocessing import Process, Event
from loguru import logger
import os

import pandas as pd
from sklearn.linear_model import LinearRegression
import opensmile
import pyaudio
import wave
import time


class Prosodie:

    def __init__(self, event):
        self._running = True
        self.event = event
        self.diff_list = []

    def terminate(self):
        self.event.set()

    def create_waveform(self, signal):
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 2
        fs = 44100  # Record at 44100 samples per second
        seconds = 5
        filename = "Prosodie.wav"

        length_queue = len(signal)

        # return no signal if its shorter than 5 seconds
        if length_queue < 215:
            return None

        p = pyaudio.PyAudio()  # Create an interface to PortAudio

        # take only the last 5 seconds (215 chunks) of the audio signal
        frames = signal[-215:]

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()
        # logger.debug("Waveform created...")

        return filename

    def loudness(self, filename, y):
        loudnessValues = y['Loudness_sma3']
        loudnessTemp = loudnessValues[filename]
        loudnessTemp = list(loudnessTemp.items())

        dataTableLoudness = {
            'start': [],
            'end': [],
            'loudness': []
        }

        for row in loudnessTemp:
            dataTableLoudness['start'].append(row[0][0])
            dataTableLoudness['end'].append(row[0][1])
            dataTableLoudness['loudness'].append(row[1])

        dfLoudness = pd.DataFrame(dataTableLoudness)

        finalLoudness = dfLoudness.tail(25)

        finalLoudness_mean = finalLoudness['loudness'].mean()
        loudness_mean = dfLoudness['loudness'].mean()

        if finalLoudness_mean <= loudness_mean:
            loudness = 1
        elif finalLoudness_mean > loudness_mean:
            loudness = 0

        return loudness

    def f0(self, filename, y):
        f0Values = y['F0semitoneFrom27.5Hz_sma3nz']
        f0Temp = f0Values[filename]
        f0Temp = list(f0Temp.items())

        dataTablef0 = {
            'start': [],
            'end': [],
            'F0': []
        }

        for row in f0Temp:
            dataTablef0['start'].append(row[0][0])
            dataTablef0['end'].append(row[0][1])
            dataTablef0['F0'].append(row[1])

        df_f0 = pd.DataFrame(dataTablef0)

        time = df_f0["end"].dt.total_seconds()
        f0 = df_f0['F0']

        x_final = time.tail(10).array.reshape(-1, 1)
        y_final = f0.tail(10)  # Die letzen 10 Werte von F0

        LR = LinearRegression().fit(x_final, y_final)
        coef = LR.coef_

        if coef[0] > 0.1:
            pitch = 1
        elif coef[0] < 0.1:
            pitch = 1
        elif coef[0].isclose(0):
            pitch = 0

        return pitch

    def slope(self, filename, y):
        slope_500_values = y['slope0-500_sma3']
        slope_1500_values = y['slope500-1500_sma3']

        slope_500_Temp = slope_500_values[filename]
        slope_500_Temp = list(slope_500_Temp.items())

        slope_1500_Temp = slope_1500_values[filename]
        slope_1500_Temp = list(slope_1500_Temp.items())

        dataTable_slope_500 = {
            'start': [],
            'end': [],
            'slope 0-500': []
        }

        for row in slope_500_Temp:
            dataTable_slope_500['start'].append(row[0][0])
            dataTable_slope_500['end'].append(row[0][1])
            dataTable_slope_500['slope 0-500'].append(row[1])

        df_slope_500 = pd.DataFrame(dataTable_slope_500)

        dataTable_slope_1500 = {
            'start': [],
            'end': [],
            'slope 500-1500': []
        }

        for row in slope_1500_Temp:
            dataTable_slope_1500['start'].append(row[0][0])
            dataTable_slope_1500['end'].append(row[0][1])
            dataTable_slope_1500['slope 500-1500'].append(row[1])

        df_slope_1500 = pd.DataFrame(dataTable_slope_1500)

        time = df_slope_500["end"].dt.total_seconds()
        x_final = time.tail(10).array.reshape(-1, 1)

        slope_500 = df_slope_500['slope 0-500']

        y_slope_500_final = slope_500.tail(10)  # Die letzen 10 Werte vom slope 0-500Hz

        LR_slope_500 = LinearRegression().fit(x_final, y_slope_500_final)
        coef_slope_500 = LR_slope_500.coef_
        intercept_slope_500 = LR_slope_500.intercept_

        slope_1500 = df_slope_1500['slope 500-1500']

        y_slope_1500_final = slope_1500.tail(10)  # Die letzen 20 Werte vom slope 0-500Hz

        LR_slope_1500 = LinearRegression().fit(x_final, y_slope_1500_final)
        coef_slope_1500 = LR_slope_1500.coef_
        intercept_slope_1500 = LR_slope_1500.intercept_
        slope = 0

        if coef_slope_500 < 0 and coef_slope_1500 != 0:
            slope = 1
        elif coef_slope_1500 > 0 and coef_slope_1500 != 0:
            slope = 0

        return slope

    def run(self, InputQueue, OutputQueue):
        logger.debug("Process enters loop")

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

        signal = []

        while not self.event.is_set():
            audio_signal = InputQueue.get()
            signal.append(audio_signal)
            filename = self.create_waveform(signal)
            # logger.debug("created audio")
            if filename is None:
                # logger.debug("audio is empty")
                continue
            audio_file = os.path.expanduser(filename)

            # Variable mit den LLD's
            y = smile.process_file(audio_file)

            loudness = self.loudness(filename, y)
            # logger.debug("loudness: {}", loudness)
            pitch = self.f0(filename, y)
            # logger.debug("pitch: {}", pitch)
            slope = self.slope(filename, y)
            # logger.debug("slope: {}", slope)

            time.sleep(0.1)

            # 0: Holding 1: Shift
            if (loudness + pitch + slope) == 3:
                shift = 1
            else:
                shift = 0

            OutputQueue.put(shift)


event = Event()
c = Prosodie(event)
t = None


def StopProsodie():
    global c, t
    # Signal termination
    c.terminate()
    logger.debug("Process terminated")


def StartProsodie(InputQueue=None, OutputQueue=None):
    global t, event, c
    if t is None:
        event = Event()
        c = Prosodie(event)
        t = Process(target=c.run, args=(InputQueue, OutputQueue))
        t.start()
        logger.debug("Process started")
