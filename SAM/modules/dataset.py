import sys
sys.path.append('/home/msuleman/ml20_scratch/fyp_galaxy')
# dirty soln for debugging
# fix relative import later

from base_dataset import RadioGalaxyNET
from transformers import SamProcessor

class SAMDataset(RadioGalaxyNET):
    def __formatOutput__(self, imgId, img, boxes, instanceMasks, labels, iscrowd, area):
        if not hasattr(self, 'processor'):
            self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        boxes = [boxes.tolist()]
        inputs = self.processor(img, input_boxes=boxes)
        inputs['ground_truth_mask'] = instanceMasks
        return inputs