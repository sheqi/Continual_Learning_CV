import random
import math
import numpy as np
from numpy.random import randint
import pdb

class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalEndCrop(object):
    """Temporally crop the given frame indices at a beginning.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[-self.size:]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out







class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        # begin_index = 32
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalUniformCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        average_duration = len(frame_indices) // self.size
        if average_duration > 0:
            out = np.multiply(list(range(self.size)), average_duration) + randint(average_duration, size=self.size)
        else:
            out = frame_indices
            for index in out:
                if len(out) >= self.size:
                    break
                out.append(index)              
        return out    





# class TemporalUniformCrop(object):
#     """Temporally crop the given frame indices at a center.
#     If the number of frames is less than the size,
#     loop the indices as many times as necessary to satisfy the size.
#     Args:
#         size (int): Desired output size of the crop.
#     """

#     def __init__(self,  skip, size):
#         self.skip = skip
#         self.size = size
#     def __call__(self, frame_indices):
#         """
#         Args:
#             frame_indices (list): frame indices to be cropped.
#         Returns:
#             list: Cropped frame indices.
#         """
#         clips = [frame_indices[i] for i in range(0, len(frame_indices), self.skip)]
#         out_clips = []
#         for clip_i_begin in clips:
#             begin_index = clip_i_begin
#             end_index = min(begin_index + self.size, len(frame_indices))
#             out = frame_indices[begin_index:end_index]
#             for index in out:
#                 if len(out) >= self.size:
#                     break
#                 out.append(index)
#             out_clips.append(out)
#             if begin_index + self.size >= len(frame_indices):
#                 break
#         # for index in out:
#         #     if len(out) >= self.size:
#         #         break
#         #     out.append(index)
#         return out_clips


# class TemporalUniformCrop(object):
#     """Temporally crop the given frame indices at a center.
#     If the number of frames is less than the size,
#     loop the indices as many times as necessary to satisfy the size.
#     Args:
#         size (int): Desired output size of the crop.
#     """

#     def __init__(self,  skip):
#         self.skip = skip
#     def __call__(self, frame_indices):
#         """
#         Args:
#             frame_indices (list): frame indices to be cropped.
#         Returns:
#             list: Cropped frame indices.
#         """
#         out = [frame_indices[i] for i in range(0, len(frame_indices), self.skip)]


#         return out