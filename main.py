import multiprocessing
from AUDIO_READ import StartAUDIO_READ, StopAUDIO_READ
from AudioChunkBroker import StartAudioChunkBroker, StopAudioChunkBroker
from VoiceActivityProjection.VAP_Model import StartVAP_Model, StopVAP_Model
import time
from loguru import logger

logger.add("main.log")

Audio_Device = "Built-in Microphone"

if __name__ == '__main__':

    with multiprocessing.Manager() as manager:
        AudioReadOutput = manager.Queue()
        AudioBrokerToVAP = manager.Queue()
        AudioBrokerToProsodieQueue = manager.Queue()
        ListAudioBrokerQueues = [AudioBrokerToVAP, AudioBrokerToProsodieQueue]
        VAPOutput = manager.Queue()

        # Audioaufzeichnung starten
        logger.debug("Start AUDIO_READ")
        StartAUDIO_READ(audiodevice=Audio_Device, OutputQueue=AudioReadOutput)

        logger.debug("Start Broker")
        StartAudioChunkBroker(AudioReadOutput, ListAudioBrokerQueues)

        logger.debug("Start VAP")
        StartVAP_Model(AudioBrokerToVAP, VAPOutput)

        time.sleep(60)

        logger.debug("Stop VAP")
        StopVAP_Model

        logger.debug("Stop Broker")
        StopAudioChunkBroker

        logger.debug("Stop AUDIO_READ")
        StopAUDIO_READ
