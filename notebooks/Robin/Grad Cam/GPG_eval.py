
from GradCam_eval import GradCamEvaluator

def run_visualization(model, input_tensor, label, method="gradcam"):
    cam_evaluator = GradCamEvaluator(model, method)
    result = cam_evaluator.evaluate(input_tensor, label)
    return result
