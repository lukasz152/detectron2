from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import cv2
import numpy as np
from scipy.signal import wiener
from skimage.restoration import denoise_nl_means

class Detector:
    def __init__(self, model="OD"):
        self.cfg = get_cfg()

        #konfig modelu
        if model == "OD":
            #model
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            #wagi modelu
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model == "IS":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model == "LVIS":
            self.cfg.merge_from_file(
                model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.DEVICE = "cpu"  # cuda
        self.predictor = DefaultPredictor(self.cfg)

    def preprocess_image(self, image):
        # Usunięcie szumów( median filtering zdj)
        processed_image = cv2.medianBlur(image, 5)
        #processed_image = cv2.GaussianBlur(image, (5, 5), 0)
        #processed_image = cv2.bilateralFilter(image, 9, 75, 75)   # filtr bilateralny  jeśli chcesz zachować detale (dokladnosc ), ale jednocześnie zmniejszyć szum.
        #from scipy.signal import wiener
        #from skimage.restoration import denoise_nl_means
        #processed_image = wiener(image) #Filtr Wienera
        #processed_image  = denoise_nl_means(image, h=0.8 * np.std(image))  # ogromnie wiekszy czas x 100 nieoplacalny !
        return processed_image

    def detect_objects(self, image):
        # Wykrywanie obiektów
        predictions = self.predictor(image)
        return predictions

    def get_detected_objects_data(self, image_path):
        image = cv2.imread(image_path)
        predictions = self.detect_objects(image)
        instances = predictions["instances"].to("cpu")
        #ramki
        bounding_boxes = instances.pred_boxes.tensor.numpy()
        #klasa
        class_labels = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        return bounding_boxes, class_labels, scores
    def visualize_predictions(self, image, predictions):
        # Wizualizacja
        viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                         instance_mode=ColorMode.IMAGE_BW)
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        return output.get_image()[:, :, ::-1]

    def on_image(self, image_path):
        # Wczytanie obrazu
        image = cv2.imread(image_path)
        # Przetworzenie
        processed_image = self.preprocess_image(image)
        # Wykrycie
        predictions = self.detect_objects(processed_image)
        # Wizualizacja
        result_image = self.visualize_predictions(processed_image, predictions)

        cv2.imshow("Result", result_image)
        cv2.waitKey(0)