from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_florence_2_segmentation.infer_florence_2_segmentation_process import InferFlorence2SegmentationParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
from torch.cuda import is_available

# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferFlorence2SegmentationWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferFlorence2SegmentationParam()
        else:
            self.parameters = param

        self.initial_model_name = self.parameters.model_name
        self.initial_cuda = self.parameters.cuda

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Cuda
        self.check_cuda = pyqtutils.append_check(
            self.grid_layout, "Cuda", self.parameters.cuda and is_available())
        self.check_cuda.setEnabled(is_available())

        # Model name
        self.combo_model = pyqtutils.append_combo(
            self.grid_layout, "Model name")
        self.combo_model.addItem("microsoft/Florence-2-base")
        self.combo_model.addItem("microsoft/Florence-2-large")
        self.combo_model.addItem("microsoft/Florence-2-base-ft")
        self.combo_model.addItem("microsoft/Florence-2-large-ft")

        self.combo_model.setCurrentText(self.parameters.model_name)

        # Task Prompt
        self.combo_task_prompt = pyqtutils.append_combo(
            self.grid_layout, "Task prompt")
        self.combo_task_prompt.addItem("REFERRING_EXPRESSION_SEGMENTATION")
        self.combo_task_prompt.addItem("REGION_TO_SEGMENTATION")

        self.combo_task_prompt.setCurrentText(self.parameters.task_prompt)

        # Prompt
        self.edit_prompt = pyqtutils.append_edit(self.grid_layout, "Prompt", self.parameters.prompt)

        # Max new tokens
        self.spin_max_new_tokens = pyqtutils.append_spin(
                                            self.grid_layout,
                                            "Max tokens",
                                            self.parameters.max_new_tokens
        )

        # New beams
        self.spin_num_beams = pyqtutils.append_spin(
                                            self.grid_layout,
                                            "Number of beams",
                                            self.parameters.num_beams
        )

        # Do sample
        self.check_do_sample = pyqtutils.append_check(
            self.grid_layout, "Do sample", self.parameters.do_sample)

        # Early stopping
        self.check_early_stopping = pyqtutils.append_check(
            self.grid_layout, "Early stopping", self.parameters.early_stopping)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot
        # Check param update
        new_model_name = self.combo_model.currentText()
        new_cuda = self.check_cuda.isChecked()
        if new_model_name != self.initial_model_name or new_cuda != self.initial_cuda:
            self.parameters.update = True
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.prompt = self.edit_prompt.text()
        self.parameters.task_prompt = self.combo_task_prompt.currentText()
        self.parameters.max_new_tokens = self.spin_max_new_tokens.value()
        self.parameters.num_beams = self.spin_num_beams.value()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.do_sample = self.check_do_sample.isChecked()
        self.parameters.early_stopping = self.check_early_stopping.isChecked()

        # Send signal to launch the algorithm main function
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferFlorence2SegmentationWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_florence_2_segmentation"

    def create(self, param):
        # Create widget object
        return InferFlorence2SegmentationWidget(param, None)
