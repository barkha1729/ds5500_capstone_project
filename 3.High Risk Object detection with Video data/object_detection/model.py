import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class Model():
    
    def __init__(self,num_classes,model_name):
        self.num_classes=num_classes
        self.model_name=model_name
    
    def create_model(self):
        
        # load Faster RCNN pre-trained model
        if(self.model_name=="faster_rcnn"):
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            
            # get the number of input features 
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # define a new head for the detector with required number of classes
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes) 
            return model