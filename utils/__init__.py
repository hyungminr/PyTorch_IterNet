import os
import numpy as np
import matplotlib.pyplot as plt

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def sec2time(sec, n_msec=0):
    """ Convert seconds to 'D days, HH:MM:SS.FFF' """
    if hasattr(sec, '__len__'):
        return [sec2time(s) for s in sec]
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if n_msec > 0:
        pattern = '%%02d:%%02d:%%0%d.%df' % (n_msec+3, n_msec)
    else:
        pattern = r'%02d:%02d:%02d'
    if d == 0:
        return pattern % (h, m, s)
    return ('%d days, ' + pattern) % (d, h, m, s)

def tensor2np(tensor):
    return np.transpose(tensor.cpu().data.numpy(), (1, 2, 0))

def img_save(x, gt, y, name='out.jpg'):
    fig = plt.figure()
    for i, img in enumerate([x, gt, y]):
        img = img.cpu().detach()
        img = np.transpose(img[0], (1, 2, 0))
        _, _, d = img.shape
        plt.subplot(1,3,i+1)
        if d == 3:
            plt.imshow(img)
        else:
            plt.imshow(img[:,:,0], cmap='gray')
    plt.tight_layout()
    fig.savefig(name)
    plt.close('all')
