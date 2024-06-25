from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_florence_2_segmentation.infer_florence_2_segmentation_process import InferFlorence2SegmentationFactory
        return InferFlorence2SegmentationFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_florence_2_segmentation.infer_florence_2_segmentation_widget import InferFlorence2SegmentationWidgetFactory
        return InferFlorence2SegmentationWidgetFactory()
