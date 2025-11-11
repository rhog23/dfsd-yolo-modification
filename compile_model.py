from ultralytics.nn.tasks import ClassificationModel
from torchsummary import summary

model = ClassificationModel("yolo12s-ghost-cls.yaml")
# summary(model, (3, 224, 224)) 

# Count GhostConv layers
ghost_count = 0
total_conv = 0

for name, module in model.model.named_modules():
    if 'ghost' in name.lower() or 'Ghost' in type(module).__name__:
        ghost_count += 1
        print(f"Found: {name} -> {type(module).__name__}")
    if 'conv' in type(module).__name__.lower():
        total_conv += 1

print(f"\nGhost modules: {ghost_count}")
print(f"Total conv-like modules: {total_conv}")
print(f"Replacement percentage: {ghost_count/total_conv*100:.1f}%")