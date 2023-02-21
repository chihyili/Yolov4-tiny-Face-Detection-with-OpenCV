import cv2
import time
import numpy as np

class Yolo:

    def __init__(self, model_path,config_file,label):
        self.thresh = .25
        self.LABELS = label#open(label_file).read().strip().split("\n")
        self.weights = model_path
        self.config_file = config_file
        self.net = cv2.dnn.readNetFromDarknet(self.config_file, self.weights)
        np.random.seed(4)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),dtype="uint8")
        self.IMG_SIZE = 416


    def run(self,image):
        image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        ln = self.net.getLayerNames()
        ln = [ln[[i][0] - 1] for i in self.net.getUnconnectedOutLayers()]


        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        end = time.time()


        print("[INFO] took {:.6f} seconds".format(end - start))

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.thresh:
            
                    box = detection[0:4] * np.array([self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.thresh, self.thresh)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in self.COLORS[classIDs[i]]]

                cv2.rectangle(image, (x, y), (x + w, y + h), color, 5)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
        
        return image

