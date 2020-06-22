import torch
import os
from abc import ABC
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision


class Interpolator(ABC):

    def __init__(self, model):
        print("Setting up interpolator")
        self.transfer_network = model
        self.transfer_network.eval()
        print(" Ready")

    ## TODO: adapt this function for multiple inputs
    def render_interpolated_image(self, interpolated_style_parameters, test_image_tensor, I_prev, style_idx=0):
        set_style_parameters(self.transfer_network, interpolated_style_parameters, style_idx)
        return self.transfer_network(test_image_tensor, I_prev, style_idx)


class TwoStyleInterpolator(Interpolator):

    def __init__(self, model):
        super(TwoStyleInterpolator, self).__init__(model)

    def interpolate(self, style_parameters_a, style_parameters_b, alpha):
        interpolated_weights = []
        interpolated_biases = []
        weights_a, biases_a = style_parameters_a
        weights_b, biases_b = style_parameters_b
        for i in range(len(weights_a)):
            interpolated_weights.append(weights_a[i].mul(1 - alpha) + weights_b[i].mul(alpha))
            interpolated_biases.append(biases_a[i].mul(1 - alpha) + biases_b[i].mul(alpha))
        return interpolated_weights, interpolated_biases

    def run_interpolation(self, style_num_a, style_num_b, step=0.25):
        assert (1/step).is_integer()
        print('Interpolating styles %d and %d with step size %.2f' % (style_num_a, style_num_b, step))

        style_parameters_a = get_style_parameters(self.transfer_network, style_num_a)
        style_parameters_b = get_style_parameters(self.transfer_network, style_num_b)

        interpolated_style_parameters_list = []
        alpha = 0
        while alpha <= 1:
            interpolated_style_parameters = self.interpolate(style_parameters_a, style_parameters_b, alpha)
            interpolated_style_parameters_list.append(interpolated_style_parameters)
            alpha += step

        return interpolated_style_parameters_list

    def produce_interpolated_grid(self, interpolated_style_parameters_list, test_image_tensor, I_prev, level, grid_dim=5):
        print('Rendering images into grid')
        output_images = []
        for interpolated_style_parameters in interpolated_style_parameters_list:
            output_images.append(self.render_interpolated_image(interpolated_style_parameters, test_image_tensor, I_prev))

        interpolation_dir = '../data/interpolation/'
        if not os.path.exists(interpolation_dir):
            os.makedirs(interpolation_dir)
        path = os.path.join(interpolation_dir + str(level) + '_two.png')
        print(' Saving to', path)
        save_tensors_as_grid(output_images, path, nrow=len(output_images))
        plot_image_tensor(load_image_as_tensor(path, transform=transforms.ToTensor()))
        print(' Done')



#### Getting and setting the weights and biases of the instance normalization layers ################

def get_all_conditional_norms(model):
    conditional_norms = []
    for name, layer in model.named_children():
        classname = layer.__class__.__name__
        if classname == 'InstanceNorm2d':
            conditional_norms.append(layer)

    return conditional_norms


def get_style_parameters(model, style_idx):
    conditional_norms = get_all_conditional_norms(model)
    weight_tensors = []
    bias_tensors = []
    for conditional_norm in conditional_norms:
        weight_tensors.append(conditional_norm.norm2ds[style_idx].weight)
        bias_tensors.append(conditional_norm.norm2ds[style_idx].bias)
    return weight_tensors, bias_tensors


def set_style_parameters(model, style_parameters, style_idx):
    """
    Load a set of style parameters into a particular style slice.
    """
    conditional_norms = get_all_conditional_norms(model)
    weight_tensors, bias_tensors = style_parameters
    for i in range(len(conditional_norms)):
        conditional_norms[i].norm2ds[style_idx].weight.data = weight_tensors[i]
        conditional_norms[i].norm2ds[style_idx].bias.data = bias_tensors[i]


##### image handling functions

# Consistent transform to scale image to 256 x 256
transform_256 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])


def load_image_as_tensor(image_path, transform=transform_256):
    """
    Load a single image as a tensor from the given path. Applies a transformation if one is given.
    """
    image = Image.open(image_path)
    image = transform(image).float()
    image = Variable(image, requires_grad=False)

    # Deal with greyscale
    if not (len(image) == 3):
        _image = torch.zeros((3, image.shape[1], image.shape[2]))
        _image[0] = image[0]
        _image[1] = image[0]
        _image[2] = image[0]
        image = _image

    return image


def save_tensor_as_image(tensor, image_path):
    """
    Save a single 3D tensor to the given image path
    """
    #torchvision.utils.save_image(tensor, image_path)
    img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(image_path)


def save_tensors_as_grid(tensors, image_path, nrow, cwidth=256, cheight=256):
    """
    Save a list of image tensors to the given image path as a grid
    """
    reshaped_tensors = []
    for tensor in tensors:
        #reshaped_tensors.append(tensor.view((3, cwidth, cheight)))
        reshaped_tensors.append(tensor.squeeze())
    torchvision.utils.save_image(reshaped_tensors, image_path, nrow=nrow, padding=10, normalize=True,
                                 scale_each=True, pad_value=255)


def plot_image_tensor(image_tensor):
    """
    Plot a single image using matplotlib.
    """
    plt.figure()
    plt.imshow(image_tensor.permute(1, 2, 0))
    plt.show()