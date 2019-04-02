try:
    import visdom
except ModuleNotFoundError:
    print('VisdomBoard ERROR: visdom package not found')
    print('You can install it following these instructions: https://github.com/facebookresearch/visdom#setup')

from .manager import get_visdom_manager
