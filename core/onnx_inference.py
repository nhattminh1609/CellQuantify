import logging
import onnxruntime as ort

logger = logging.getLogger("CellCounter.ONNX")


class ONNXModel:
    def __init__(self, model_path):
        try:
            self.session = ort.InferenceSession(model_path)

            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            self.output_name = self.session.get_outputs()[0].name

            shape = input_info.shape
            self.input_h = shape[2] if isinstance(shape[2], int) else 640
            self.input_w = shape[3] if isinstance(shape[3], int) else 640

            logger.info("Loaded ONNX model: %s", model_path)

        except Exception:
            logger.exception("Failed to load ONNX model: %s", model_path)
            raise

    def get_input_size(self):
        return self.input_w, self.input_h

    def inference(self, input_tensor):
        try:
            return self.session.run(
                [self.output_name],
                {self.input_name: input_tensor}
            )[0]
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            raise