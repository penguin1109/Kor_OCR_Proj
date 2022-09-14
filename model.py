"""
(1) Feature Transformation 
    - 예를 들면 휘어 있다거나 굴곡이 있는 등의 형태적으로 완벽한 일직선이 아닌 경우에는 alignment를 조절해야 할수 있다.
(2) Feature Extraction
    - VGG
    - RCNN
    - ResNet
(3) Sequence Modeling
    - BiLSTM
(4) Prediction
    - CTC
    - Attention
"""

import torch
import torch.nn as nn
import torch.nn.fucntional as F

from arches.feature_extractor import *
from arches.sequence_modeling import *
from arches.predictior import *

class Model(nn.Module):
    def __init__(self, 
                 input_channel,
                 output_channel,
                 hidden_size,
                 num_class,
                 pred_model = 'att'
                 ):
        super(Model, self).__init__()
        ## Feature Extraction ##
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel # int(imgH / 16 - 1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1)) # Final Transformation: (imgH / 15 - 1) -> 1
        
        ## Sequence Modeling ##
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        )
        self.SequenceModeling_output = hidden_size
        
        ## Prediction ##
        if pred_model == 'att':
            self.prediction = Attention(self.SequenceModeling_output, hidden_size, num_class)
        else:
            self.prediction = nn.Linear(self.SequenceModeling_output, num_class) # CTC
        
        
    def forward(self, x):
        ## Feature Extraction Stage ##
        visual_feature = self.FeatureExtraction(x)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2)) # [B, C, H, W] -> [B, W, C, H]
        visual_feature = visual_feature.squeeze(0)
        
        ## Sequence Modeling Stage ##
        contextual_feature = self.SequenceModeling(visual_feature)
        
        ## Prediction Stage ##
        prediction = self.Prediction(contextual_feature.contiguous())
        
        return prediction
        
        
        