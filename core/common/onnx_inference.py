import logging
import onnxruntime as ort

import config

logger = logging.getLogger("CellCounter.ONNX")


class ONNXModel:
    def __init__(self, model_path, providers=None):
        try:
            if providers is None:
                # Prefer GPU when available, always keep CPU as fallback.
                available = ort.get_available_providers()
                preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
                providers = [p for p in preferred if p in available] or available
            self.session = ort.InferenceSession(model_path, providers=providers)

            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            self.output_name = self.session.get_outputs()[0].name

            shape = input_info.shape
            default = config.DEFAULT_MODEL_INPUT_SIZE
            self.input_h = shape[2] if isinstance(shape[2], int) else default
            self.input_w = shape[3] if isinstance(shape[3], int) else default

            logger.info("Loaded ONNX model: %s (providers=%s)", model_path, self.session.get_providers())

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
        except Exception:
            logger.exception("Inference error")
            raise
