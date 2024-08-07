import copy
import torch
import os
import cv2
import numpy as np
from ikomia import core, dataprocess, utils
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw

# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferFlorence2SegmentationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = 'microsoft/Florence-2-large'
        self.task_prompt = 'REFERRING_EXPRESSION_SEGMENTATION'
        self.prompt = 'a green car'
        self.max_new_tokens = 1024
        self.num_beams = 3
        self.do_sample = False
        self.early_stopping = False
        self.cuda = torch.cuda.is_available()
        self.update = False

    def set_values(self, params):
        # Set parameters values from Ikomia Studio or API
        self.update = utils.strtobool(params["cuda"]) != self.cuda or self.model_name != str(params["model_name"])
        self.model_name = str(params["model_name"])
        self.task_prompt = str(params["task_prompt"])
        self.prompt = str(params["prompt"])
        self.max_new_tokens = int(params["max_new_tokens"])
        self.num_beams = int(params["num_beams"])
        self.do_sample = utils.strtobool(params["do_sample"])
        self.early_stopping = utils.strtobool(params["early_stopping"])
        self.cuda = utils.strtobool(params["cuda"])

    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        params = {}
        params["model_name"] = str(self.model_name)
        params["task_prompt"] = str(self.task_prompt)
        params["prompt"] = str(self.prompt)
        params["max_new_tokens"] = str(self.max_new_tokens)
        params["num_beams"] = str(self.num_beams)
        params["do_sample"] = str(self.do_sample)
        params["early_stopping"] = str(self.early_stopping)
        params["cuda"] = str(self.cuda)

        return params


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferFlorence2Segmentation(dataprocess.CInstanceSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CInstanceSegmentationTask.__init__(self, name)
        # Create parameters object
        if param is None:
            self.set_param_object(InferFlorence2SegmentationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.processor = None
        self.model = None
        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
        self.device = torch.device("cpu")

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def load_model(self, param):
        try:
            self.processor = AutoProcessor.from_pretrained(
                                    param.model_name,
                                    cache_dir=self.model_folder,
                                    local_files_only=True,
                                    trust_remote_code=True
                                    )

            self.model = AutoModelForCausalLM.from_pretrained(
                                    param.model_name,
                                    cache_dir=self.model_folder,
                                    local_files_only=True,
                                    trust_remote_code=True
                                    ).eval()

        except Exception as e:
            print(f"Failed with error: {e}. Trying without the local_files_only parameter...")
            self.processor = AutoProcessor.from_pretrained(
                                        param.model_name,
                                        cache_dir=self.model_folder,
                                        trust_remote_code=True
                                        )

            self.model = AutoModelForCausalLM.from_pretrained(
                                    param.model_name,
                                    cache_dir=self.model_folder,
                                    trust_remote_code=True
                                    ).eval()
        self.model.to(self.device)


    def infer(self, task_prompt, img, param, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        img_h, img_w = img.shape[:2]

        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.device)

        # Inference
        generated_ids = self.model.generate(
                                    input_ids=inputs["input_ids"],
                                    pixel_values=inputs["pixel_values"],
                                    max_new_tokens=param.max_new_tokens,
                                    early_stopping=param.early_stopping,
                                    do_sample=param.do_sample,
                                    num_beams=param.num_beams,
                                    )
        generated_text = self.processor.batch_decode(
                                            generated_ids,
                                            skip_special_tokens=False
                                            )[0]
        parsed_answer = self.processor.post_process_generation(
                                            generated_text,
                                            task=task_prompt,
                                            image_size=(img_w, img_h)
                                            )

        return parsed_answer


    def generate_binary_mask_and_bounding_box(self, results, image_shape):
        # Initialize binary mask with uint8
        binary_mask = Image.new('L', (image_shape[1], image_shape[0]), 0)

        draw = ImageDraw.Draw(binary_mask)

        # Extract polygon coordinates and fill polygons in one go
        polygons = results['polygons']
        for polygon in polygons:
            for _polygon in polygon:
                _polygon = np.array(_polygon).reshape(-1, 2)
                if len(_polygon) < 3:
                    print('Invalid polygon:', _polygon)
                    continue
                _polygon = _polygon.reshape(-1).tolist()
                # Draw the polygon
                draw.polygon(_polygon, outline=1, fill=1)

        # Convert binary mask back to numpy array
        binary_mask = np.array(binary_mask, dtype=np.uint8)

        # Find bounding box using numpy operations
        all_pts_concat = np.concatenate([np.array(polygon).reshape(-1, 2) for polygon in polygons])
        x_coords = all_pts_concat[:, 0]
        y_coords = all_pts_concat[:, 1]
        x1, y1 = np.min(x_coords), np.min(y_coords)
        x2, y2 = np.max(x_coords), np.max(y_coords)
        h = y2 - y1
        w = x2 - x1

        # Create the bounding box as a tuple
        bounding_box = (x1, y1, w, h)

        return binary_mask, bounding_box

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters
        param = self.get_param_object()

        # Get input :
        input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = input.get_image()

        # Image pre-process
        shape_img = src_image.shape[:2]

        # Load model
        if param.update or self.model is None:
            self.device = torch.device(
                "cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
            self.load_model(param)
            param.update = False

        task_prompt_formatted = f'<{param.task_prompt}>'

        # Inference
        with torch.no_grad():
            output = self.infer(task_prompt_formatted, src_image,  param, param.prompt)

        results = output[task_prompt_formatted]

        mask, bbox = self.generate_binary_mask_and_bounding_box(results, shape_img)

        # Set classe names
        self.set_names([param.prompt])
        self.add_object(
            0,
            0,
            int(0),
            float(1),
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
            mask
        )

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferFlorence2SegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_florence_2_segmentation"
        self.info.short_description = "Run florence 2 segmentation with or without text prompt"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Instance Segmentation"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "B. Xiao, H. Wu, W. Xu, X. Dai, H. Hu, Y. Lu, M. Zeng, C. Liu, L. Yuan"
        self.info.article = "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks"
        self.info.journal = "arXiv:2311.06242"
        self.info.year = 2023
        self.info.license = "MIT License"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_florence_2_caption"
        self.info.original_repository = "https://github.com/googleapis/python-vision"
        # Python version
        self.info.min_python_version = "3.10.0"
        # Keywords used for search
        self.info.keywords = "Florence,Microsoft,Segmentation,Unified,Pytorch"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "INSTANCE_SEGMENTATION"
        self.info.os = utils.OSType.LINUX

    def create(self, param=None):
        # Create algorithm object
        return InferFlorence2Segmentation(self.info.name, param)