import multiprocessing
from Audio_Video_Read.AUDIO_READ import StartAUDIO_READ, StopAUDIO_READ
from Broker.AudioChunkBroker import StartAudioChunkBroker, StopAudioChunkBroker
from VoiceActivityProjection.VAP_Model import StartVAP_Model, StopVAP_Model
from Headpose.HeadposeDetection import StartHeadposeDetection, StopHeadposeDetection
from Processing.Processing import StartProcessing, StopProcessing
from Prosodie.Prosodie import StartProsodie, StopProsodie
from Processing.SendToSkill import StartSending, StopSending
import time
from loguru import logger

logger.add("main.log")

# define the name of the microphone
Audio_Device = "Mikrofon (Jabra EVOLVE 20 SE)"

if __name__ == '__main__':
    print("Available CPU cores: ", multiprocessing.cpu_count())

    # start manager for multiprocessing
    with multiprocessing.Manager() as manager:
        AudioReadOutput = manager.Queue()
        AudioBrokerToVAP = manager.Queue()
        AudioBrokerToProsodie = manager.Queue()
        ListAudioBrokerQueues = [AudioBrokerToVAP, AudioBrokerToProsodie]
        HeadposeOutput = manager.Queue()
        VAPOutput = manager.Queue()
        ProsodieOutput = manager.Queue()
        ListProcessingQueues = [VAPOutput, ProsodieOutput, HeadposeOutput]
        ProcessingOutput = manager.Queue()

        # start recording
        logger.debug("Start AUDIO_READ")
        StartAUDIO_READ(audiodevice=Audio_Device, OutputQueue=AudioReadOutput)

        # start the broker
        logger.debug("Start Broker")
        StartAudioChunkBroker(AudioReadOutput, ListAudioBrokerQueues)

        # start the VAP Model
        logger.debug("Start VAP")
        StartVAP_Model(AudioBrokerToVAP, VAPOutput)

        # start Prosodie
        logger.debug("Start Prosodie")
        StartProsodie(InputQueue=AudioBrokerToProsodie, OutputQueue=ProsodieOutput)

        # start processing of the video
        logger.debug("Start HeadposeDetection")
        StartHeadposeDetection(InputQueue=None, OutputQueue=HeadposeOutput)

        # start Processing
        logger.debug("Start Processing")
        StartProcessing(ListProcessingQueues, ProcessingOutput)

        # start Send to Skill
        logger.debug("Start sending to skill")
        StartSending(InputQueue=ProcessingOutput)

        time.sleep(500)  # This defines how long the system is active in seconds

        logger.debug("Stop sending to Skill")
        StopSending()

        logger.debug("Stop Prosodie")
        StopProsodie()

        logger.debug("Stop VAP")
        StopVAP_Model()

        logger.debug("Stop Broker")
        StopAudioChunkBroker()

        logger.debug("Stop AUDIO_READ")
        StopAUDIO_READ()

        logger.debug("Stop Processing")
        StopProcessing()

        logger.debug("Stop HeadposeDetection")
        StopHeadposeDetection()

        time.sleep(10)
