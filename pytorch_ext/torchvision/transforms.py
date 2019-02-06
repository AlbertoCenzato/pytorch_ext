from typing import Callable, Union, Tuple, List
from functools import singledispatch

import cv2
import torch as th
import numpy as np
import PIL


TensorOrArray = Union[th.Tensor, np.ndarray]
TensorList = List[th.Tensor]
ArrayList  = List[np.ndarray]


class PILToNumpy:

    def __call__(self, image: PIL.Image) -> np.ndarray:
        return np.array(image)


@singledispatch
def swapaxes(tensor: th.Tensor, dim0: int, dim1: int) -> th.Tensor:
    return th.transpose(tensor, dim0, dim1)


@swapaxes.register
def swapaxes_np(array: np.ndarray, dim0: int, dim1: int) -> np.ndarray:
    return np.swapaxes(array, dim0, dim1)


@singledispatch
def get_dim(tensor: th.Tensor, axis: int) -> th.Tensor:
    return tensor.size(axis)


@get_dim.register
def get_dim_np(array: np.ndarray, axis: int) -> np.ndarray:
    return array.shape[axis]


# NOTE: currently I've not found a way to single dispatch on
#       composed types such as List[th.Tensor] or List[np.ndarray]
#       not even creating a new type with 
#       typing.NewType('TensorList', typing.List[torch.Tensor])
def stack(tensor_list: Union[TensorList, ArrayList], axis: int) -> TensorOrArray:
    if isinstance(tensor_list[0], th.Tensor):
        return th.stack(tensor_list, axis)
    else:
        return np.stack(tensor_list, axis)


class RepeatAlongAxis:
    """
        Applies the given transform to the input tensor iterating
        along the specified dimension. Can handle both numpy.ndarray
        and torch.Tensor.
    """

    def __init__(self, transform: Callable, axis: int):
        self.transform = transform
        self.axis = axis

    def __call__(self, tensor: TensorOrArray) -> TensorOrArray:
        transposed = swapaxes(tensor, 0, self.axis)
        resulting_tensors = []
        for i in range(get_dim(transposed, 0)):
            transformed = self.transform(transposed[i,:])
            resulting_tensors.append(transformed)
        stacked = stack(resulting_tensors, 0)
        return swapaxes(stacked, 0, self.axis)

    
class CutSequence:

    def __init__(self, begin, end):
        self.begin = begin
        self.end   = end

    def __call__(self, tensor: TensorOrArray) -> TensorOrArray:
        return tensor[self.begin:self.end, :]


class BlurImage:

    def __init__(self, kernel_shape: Tuple[int, int], sigma: float):
        self.kernel_shape = kernel_shape
        self.sigma = sigma

    def __call__(self, image: np.ndarray) -> np.ndarray:
        blurred_image = cv2.GaussianBlur(image, self.kernel_shape, self.sigma)
        if len(image.shape) == 3:
            blurred_image = np.reshape(blurred_image, blurred_image.shape + (1,))
        return blurred_image
