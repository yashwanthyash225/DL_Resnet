import torch
from torch.functional import Tensor
import torch.nn as nn

""" This script defines the network.
"""

class ResNet(nn.Module):
    def __init__(self,
            resnet_version,
            resnet_size,
            num_classes,
            first_num_filters,
        ):
        """
        1. Define hyperparameters.
        Args:
            resnet_version: 1 or 2, If 2, use the bottleneck blocks.
            resnet_size: A positive integer (n).
            num_classes: A positive integer. Define the number of classes.
            first_num_filters: An integer. The number of filters to use for the
                first block layer of the model. This number is then doubled
                for each subsampling block layer.
        
        2. Classify a batch of input images.

        Architecture (first_num_filters = 16):
        layer_name      | start | stack1 | stack2 | stack3 | output      |
        output_map_size | 32x32 | 32X32  | 16x16  | 8x8    | 1x1         |
        #layers         | 1     | 2n/3n  | 2n/3n  | 2n/3n  | 1           |
        #filters        | 16    | 16(*4) | 32(*4) | 64(*4) | num_classes |

        n = #residual_blocks in each stack layer = self.resnet_size
        The standard_block has 2 layers each.
        The bottleneck_block has 3 layers each.
        
        Example of replacing:
        standard_block      conv3-16 + conv3-16
        bottleneck_block    conv1-16 + conv3-16 + conv1-64

        Args:
            inputs: A Tensor representing a batch of input images.
        
        Returns:
            A logits Tensor of shape [<batch_size>, self.num_classes].
        """
        super(ResNet, self).__init__()
        self.resnet_version = resnet_version
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

        ### YOUR CODE HERE
        # define conv1
        self.start_layer = nn.Conv2d(in_channels=3, out_channels=first_num_filters, kernel_size=3, padding=1)
        ### YOUR CODE HERE

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.
        if self.resnet_version == 1:
            self.batch_norm_relu_start = batch_norm_relu_layer(
                num_features=self.first_num_filters, 
                eps=1e-5, 
                momentum=0.997,
            )
        if self.resnet_version == 1:
            block_fn = standard_block
        else:
            block_fn = bottleneck_block

        self.stack_layers = nn.ModuleList()
        temp_filters = self.first_num_filters
        for i in range(3):
            filters = self.first_num_filters * (2**i)
            strides = 1 if i == 0 else 2
            self.stack_layers.append(stack_layer(filters, block_fn, strides, self.resnet_size, temp_filters))
            temp_filters = filters if resnet_version == 1 else filters*4
        self.output_layer = output_layer(temp_filters, self.resnet_version, self.num_classes)
    
    def forward(self, inputs):
        outputs = self.start_layer(inputs)
        if self.resnet_version == 1:
            outputs = self.batch_norm_relu_start(outputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs

#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.997): 
        super(batch_norm_relu_layer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum)
        
    def forward(self, inputs: Tensor):
        temp = self.bn(inputs)
        temp = torch.relu(temp)
        return temp


class standard_block(nn.Module):
    """ Creates a standard residual block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters):
        super(standard_block, self).__init__()
        ### YOUR CODE HERE
        self.projection_shortcut = projection_shortcut
        self.cnn1 = nn.Conv2d(in_channels=first_num_filters, out_channels=filters, stride=strides, kernel_size=3, padding=1)
        self.bn_relu_1 = batch_norm_relu_layer(num_features=filters, eps=1e-5, momentum=0.997)
        self.cnn2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=filters, eps=1e-5, momentum=0.997)
        ### YOUR CODE HERE

    def forward(self, inputs: Tensor):
        ### YOUR CODE HERE
        temp = self.cnn1(inputs)
        temp = self.bn_relu_1(temp)
        temp = self.cnn2(temp)
        temp = self.bn2(temp)
        temp = temp + self.projection_shortcut(inputs)
        temp = torch.relu(temp)
        return temp
        ### YOUR CODE HERE

class bottleneck_block(nn.Module):
    """ Creates a bottleneck block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution. NOTE: filters_out will be 4xfilters.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters):
        super(bottleneck_block, self).__init__()

        ### YOUR CODE HERE
        # Hint: Different from standard lib implementation, you need pay attention to 
        # how to define in_channel of the first bn and conv of each block based on
        # Args given above.
        self.bn_relu_1 = batch_norm_relu_layer(num_features=first_num_filters, eps=1e-5, momentum=0.997)
        self.cnn1 = nn.Conv2d(in_channels=first_num_filters, out_channels=filters, kernel_size=1, stride=strides)
        self.bn_relu_2 = batch_norm_relu_layer(num_features=filters, eps=1e-5, momentum=0.997)
        self.cnn2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)
        self.bn_relu_3 = batch_norm_relu_layer(num_features=filters, eps=1e-5, momentum=0.997)
        self.cnn3 = nn.Conv2d(in_channels=filters, out_channels=filters*4, kernel_size=1)
        self.projection_shortcut = projection_shortcut
        ### YOUR CODE HERE
    
    def forward(self, inputs: Tensor):
        temp = self.bn_relu_1(inputs)
        temp = self.cnn1(temp)
        temp = self.bn_relu_2(temp)
        temp = self.cnn2(temp)
        temp = self.bn_relu_3(temp)
        temp = self.cnn3(temp)
        temp = temp + self.projection_shortcut(inputs)
        return temp
        ### YOUR CODE HERE
        # The projection shortcut should come after the first batch norm and ReLU
		# since it performs a 1x1 convolution.
        ### YOUR CODE HERE

class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
			    convolution in a block.
		block_fn: 'standard_block' or 'bottleneck_block'.
		strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
        resnet_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters):
        super(stack_layer, self).__init__()
        filters_out = filters * 4 if block_fn is bottleneck_block else filters
        ### END CODE HERE
        # Only the first block per stack_layer uses projection_shortcut and strides
        self.blocks_list = nn.ModuleList()
        for i in range(resnet_size):
            if block_fn == standard_block:
                if i == 0 and strides > 1:
                    projection_shortcut = nn.Conv2d(in_channels=first_num_filters, out_channels=filters_out, kernel_size=1, stride=strides)
                    temp_stride = strides
                    temp_filters = first_num_filters
                else:
                    projection_shortcut = lambda x : x
                    temp_stride = 1
                    temp_filters = filters
            else:
                if i == 0:
                    projection_shortcut = nn.Conv2d(in_channels=first_num_filters, out_channels=filters_out, kernel_size=1, stride=strides)
                    temp_stride = strides
                    temp_filters = first_num_filters
                else:
                    projection_shortcut = lambda x : x
                    temp_stride = 1
                    temp_filters = filters * 4
            self.blocks_list.append(block_fn(filters, projection_shortcut, temp_stride, temp_filters))
        ### END CODE HERE
    
    def forward(self, inputs: Tensor):
        ### END CODE HERE
        temp = inputs
        for block in self.blocks_list:
            temp = block(temp)
        return temp
        ### END CODE HERE

class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """
    def __init__(self, filters, resnet_version, num_classes):
        super(output_layer, self).__init__()
        # Only apply the BN and ReLU for model that does pre_activation in each
		# bottleneck block, e.g. resnet V2.
        if (resnet_version == 2):
            self.bn_relu = batch_norm_relu_layer(filters, eps=1e-5, momentum=0.997)
        self.filters = filters
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(filters, num_classes)
        self.version = resnet_version
        self.soft_max = nn.Softmax(dim=1)
        ### END CODE HERE
        
        ### END CODE HERE
    
    def forward(self, inputs: Tensor):
        ### END CODE HERE
        temp = self.avg_pool(inputs)
        if self.version == 2:
            temp = self.bn_relu(temp)
        temp = temp.view(-1, self.filters)
        temp = self.fc(temp)
        temp = self.soft_max(temp)
        return temp
        ### END CODE HERE