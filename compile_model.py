from ultralytics.nn.tasks import ClassificationModel
from torchsummary import summary

model = ClassificationModel("yolo12n-ghost-cls.yaml")
summary(model, (3, 224, 224)) 