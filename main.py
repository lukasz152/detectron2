from Detector import *
import warnings
# import torch
# import time
#
# start_time = time.time()
warnings.filterwarnings("ignore", category=UserWarning)
# torch.cuda
detector=Detector(model="IS")

#instances: Instances = predictions["instances"].to("cpu")   lista wykrytych obiektow i ich nazw (dokumentacja) w trakcie

# end_time = time.time()
# execution_time = end_time - start_time
# print("Czas działania:", execution_time, "sekundy")
detector.on_image("zdj.jpg")
# end_time = time.time()
# execution_time = end_time - start_time
# print("Czas działania:", execution_time, "sekundy")


#INPUT.MIN_SIZE_TRAIN do zmiany
#INPUT.MIN_SIZE_TEST do zmiany
