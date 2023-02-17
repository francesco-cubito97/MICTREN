from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from collections import OrderedDict 

class Backbone(nn.Module):
    def __init__(self, *args):
        super().__init__(*args)
        self.output_layers = [6, 10]
        self.selected_out = OrderedDict()
        
        # Pretrained MobileNetV3 model
        self.pretrained = mobilenet_v3_small(weights = MobileNet_V3_Small_Weights.DEFAULT)
        # Remove the classification module at the end
        classifier = list(self.pretrained.classifier.children())[:-1]
        self.pretrained.classifier = nn.Sequential(*classifier) # final ouput size [batch_size, 1024]

        self.fhooks = []

        for i,layer in enumerate(list(self.pretrained.features._modules.values())):
            if i in self.output_layers:
                #print(i, dict(layer.named_modules())["block.2.avgpool"])
                self.fhooks.append(dict(layer.named_modules())["block.2.avgpool"].register_forward_hook(self.forward_hook(f"AvgPoolOuput-{i}")))
    
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        out = self.pretrained(x)
        intermediate_outputs = [list(self.selected_out.values())[i].squeeze() for i in range(len(self.output_layers))]
        intermediate_outputs = torch.cat([intermediate_outputs[0], intermediate_outputs[1]], dim=1)
        return intermediate_outputs, out