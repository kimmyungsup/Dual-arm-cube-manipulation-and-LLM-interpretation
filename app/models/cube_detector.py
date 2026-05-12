import os

import cv2 as cv
import numpy as np
import tflite_runtime.interpreter as tflite


def _load_tflite_delegate(delegate_name: str, delegate_path: str = None):
    """Load an optional TFLite delegate, returning None if unavailable."""
    if not delegate_name or delegate_name.lower() in ("cpu", "none", "off"):
        return None

    name = delegate_name.lower()
    candidate_paths = []
    if delegate_path:
        candidate_paths.append(delegate_path)
    elif name == "gpu":
        candidate_paths.extend([
            "libtensorflowlite_gpu_delegate.so",
            "libtensorflowlite_gpu_gl.so",
        ])
    else:
        candidate_paths.append(delegate_name)

    last_error = None
    for path in candidate_paths:
        try:
            delegate = tflite.load_delegate(path)
            print(f"[CUBENET][LOAD] TFLite delegate loaded: {path}")
            return delegate
        except Exception as e:
            last_error = e

    print(f"[CUBENET][WARN] failed to load TFLite delegate '{delegate_name}': {last_error}")
    print("[CUBENET][WARN] falling back to CPU TFLite interpreter")
    return None


class Detection:
    def __init__(self, position: tuple, score: float) -> None:
        self.position = position
        self.score = score

    def get_position(self, frame: np.ndarray) -> tuple[int, ...]:
        top, left, bot, right = self.position
        height, width, _ = frame.shape

        top, bot = int(top * height), int(bot * height)
        left, right = int(left * width), int(right * width)

        return top, left, bot, right

    def draw(self, frame: np.ndarray) -> None:
        top, left, bot, right = self.position
        height, width, _ = frame.shape

        top, bot = int(top * height), int(bot * height)
        left, right = int(left * width), int(right * width)

        cv.rectangle(frame, (left, top), (right, bot), (0, 255, 0), 2)


class TFLiteDetector(tflite.Interpreter):
    def __init__(
        self,
        model_path: str,
        delegate: str = None,
        delegate_path: str = None,
        num_threads: int = None,
    ) -> None:
        delegate = delegate if delegate is not None else os.environ.get("CUBENET_TFLITE_DELEGATE", "cpu")
        delegate_path = delegate_path if delegate_path is not None else os.environ.get("CUBENET_TFLITE_DELEGATE_PATH")
        num_threads = int(num_threads if num_threads is not None else os.environ.get("CUBENET_TFLITE_NUM_THREADS", "4"))

        delegates = []
        loaded_delegate = _load_tflite_delegate(delegate, delegate_path)
        if loaded_delegate is not None:
            delegates.append(loaded_delegate)

        kwargs = {"model_path": model_path, "num_threads": num_threads}
        if delegates:
            kwargs["experimental_delegates"] = delegates

        try:
            super().__init__(**kwargs)
            self.allocate_tensors()
        except Exception as e:
            if not delegates:
                raise
            print(f"[CUBENET][WARN] TFLite delegate initialization failed: {e}")
            print("[CUBENET][WARN] retrying TFLite detector on CPU")
            delegate = "cpu"
            super().__init__(model_path=model_path, num_threads=num_threads)
            self.allocate_tensors()

        self.input_details = self.get_input_details()
        self.output_details = self.get_output_details()

        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
        self.input_address = self.input_details[0]['index']
        self.delegate = delegate
        self.num_threads = num_threads

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        _, height, width, _ = self.input_shape

        image = cv.resize(image, (width, height))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        tensor = image.astype(self.input_dtype)
        tensor = (tensor - 127.5) / 127.5

        return np.expand_dims(tensor, 0)

    def _predict(self, tensor: np.ndarray) -> tuple[tuple, float]:
        self.set_tensor(self.input_address, tensor)
        self.invoke()

        position = self.get_tensor(self.output_details[1]['index'])
        score = self.get_tensor(self.output_details[0]['index'])

        return tuple(np.squeeze(position)), float(score)

    def detect(self, image: np.ndarray) -> Detection:
        tensor = self._preprocess(image)
        position, score = self._predict(tensor)

        return Detection(position, score)

    def warmup(self, runs: int = 1) -> None:
        """Run dummy inferences so first real frame does not pay allocation/JIT cost."""
        _, height, width, channels = self.input_shape
        dummy = np.zeros((height, width, channels), dtype=np.uint8)
        for _ in range(max(0, int(runs))):
            self.detect(dummy)
