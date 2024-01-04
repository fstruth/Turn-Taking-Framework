import cv2
from multiprocessing import Process, Event
from loguru import logger
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python


class Mediapipe:

    def __init__(self, event):
        self._running = True
        self.event = event

    def terminate(self):
        self.event.set()

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

        return annotated_image

    def plot_face_blendshapes_bar_graph(self, face_blendshapes):
        # Extract the face blendshapes category names and scores.
        face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
        face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
        # The blendshapes are ordered in decreasing score value.
        face_blendshapes_ranks = range(len(face_blendshapes_names))

        fig, ax = plt.subplots(figsize=(12, 12))
        bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
        ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
        ax.invert_yaxis()

        # Label each bar with values
        for score, patch in zip(face_blendshapes_scores, bar.patches):
            plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

        ax.set_xlabel('Score')
        ax.set_title("Face Blendshapes")
        plt.tight_layout()
        plt.show()

    def run(self, OutputQueue, InputQueue):

        # STEP 2: Create an FaceLandmarker object.
        base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        options = mp.tasks.vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        while not self.event.is_set():
            numpy_image = InputQueue.get()
            # Load the input image from a numpy array.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

            # STEP 4: Detect face landmarks from the input image.
            detection_result = detector.detect(mp_image)

            # STEP 5: Process the detection result. In this case, visualize it.
            annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
            cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


event = Event()
c = Mediapipe(event)
t = None


def StopMediapipe():
    global c, t
    # Signal termination
    c.terminate()
    logger.debug("Process terminated")


def StartMediapipe(InputQueue=None, OutputQueue=None):
    global t, event, c
    # c = VideoRecorder_WebCam()
    if t is None:
        event = Event()
        c = Mediapipe(event)
        t = Process(target=c.run, args=(OutputQueue, InputQueue))
        t.start()
        logger.debug("Process started")
