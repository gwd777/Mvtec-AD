import torch
import copy
import backbones

class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device, train_backbone=False):
        super(NetworkFeatureAggregator, self).__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        self.train_backbone = train_backbone

        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []

        for handle in self.backbone.hook_handles:
            handle.remove()

        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            last_layer = layers_to_extract_from[-1]
            forward_hook = ForwardHook(self.outputs, extract_layer, last_layer)

            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                backbone_modeles = backbone.__dict__["_modules"]
                network_layer = backbone_modeles[extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                net_model_layer = network_layer[-1]
                self.backbone.hook_handles.append(
                    net_model_layer.register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images, eval=True):
        self.outputs.clear()
        self.backbone(images)
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    '''
    __call__ 方法是在网络的每一层前向传播过程中调用的。当每一层的前向传播完成时，这个方法会将该层的输出存储到 self.outputs 字典中，使用层名作为键
    具体来说，ForwardHook 类中的 __call__ 方法将特定层的输出保存到 self.outputs 字典中的相应键下。这样，当整个网络的前向传播完成后，self.outputs 字典中就包含了所有指定层的输出
    '''
    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        # if self.raise_exception_to_break:
        #     raise LastLayerToExtractReachedException()
        return None

if __name__ == '__main__':
    forward_modules = torch.nn.ModuleDict({})

    layers_to_extract_from = ('layer2', 'layer3')
    device = torch.device("cuda:0")
    train_backbone = True
    backbone = backbones.load('wideresnet50')
    input_shape = (3, 288, 288)

    feature_aggregator = NetworkFeatureAggregator(
        backbone, layers_to_extract_from, device, train_backbone
    )

    # feature_dimensions = feature_aggregator.feature_dimensions(input_shape)

    forward_modules["feature_aggregator"] = feature_aggregator

    # 构造Image
    images = torch.randn(3, 3, 288, 288, dtype=torch.float32)
    images = images.to(torch.device("cuda"))

    print('__________shape___________>', images.shape)
    features = forward_modules["feature_aggregator"](images, eval=False)

    features = [features[layer] for layer in layers_to_extract_from]
