#!/usr/env/bin python
# Date created: Wed May 23 2018


# Some simple video editing utils using opencv

import pytube
import os
import sys
import cv2


class VideoReader(object):
    """
    Video reader object
    """

    def __init__(self, video_path):
        """
        :param video_path: path to the video file
        """

        self.handle = cv2.VideoCapture(video_path)
        if not self.handle.isOpened():
            sys.exit('OpenCV cannot open video {}'.format(video_path))
        self._fps = int(self.handle.get(cv2.CAP_PROP_FPS))
        self._width = int(self.handle.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.handle.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._four_cc = int(self.handle.get(cv2.CAP_PROP_FOURCC))
        self._frame_index = -1

    def getTotalNumberOfFrames(self):
        """
        :return: Total number of frames in the video file
        """

        return int(self.handle.get(cv2.CAP_PROP_FRAME_COUNT))

    def nextFrame(self):
        """
        Get the next frame of the current video reader
        :return: (ndarray) frame data
        """

        success, frame = self.handle.read()
        self._frame_index += 1
        if success:
            timeStamp = self.handle.get(cv2.CAP_PROP_POS_MSEC)
            return success, frame, timeStamp
        else:
            timeStamp = None
        return success, None, timeStamp

    def nextKthFrame(self, k):
        """
        Get the next k-th frame from the current frame index
        :param k: (int) the next k-th frame
        :return: (ndarray) frame data
        """
        # read the k-th frame from current position.
        n = 0
        result = (False, None, None)
        while n < k:
            n += 1
            result = self.nextFrame()
            if not result[0]:
                return result
        return result

    def extractPiece(self, start_frame, end_frame, name, save_dir='.'):
        """
        Extract a short piece from the original video and
        save it as a new file.
        :param start_frame: (int) start frame index
        :param end_frame:  (int) end frame index
        :param name: (str) new file name -- example.mp4
        :param save_dir: directory name before the file name
        :return: full path to the newly generated file
        """
        full_path = os.path.join(os.path.abspath(save_dir), name)

        assert start_frame < end_frame
        assert end_frame <= self.getTotalNumberOfFrames() - 1

        writer = cv2.VideoWriter(
            full_path, self._four_cc, self._fps, (self._width, self._height))

        old_index = self._frame_index
        self.handle.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self._frame_index = start_frame
        for i in range(start_frame, end_frame):
            ret, frame, stamp = self.nextFrame()
            if ret:
                writer.write(frame)
            else:
                print('Extract frame {} failed. Stop at {}th frame.'.
                      format(i, i - 1))
                break

        writer.release()
        self.handle.set(cv2.CAP_PROP_POS_FRAMES, old_index)
        self._frame_index = old_index
        print('Video piece written to {}'.format(full_path))
        return full_path

    def release(self):
        self.handle.release()

def get_video_from_youtube(video_address, fmt='mp4', save_dir=None):
    """
    Download a youtube video. Fixed 720p resolution.
    :param video_address: url to the youtube video
    :param fmt: video file format
    :param save_dir:
    :return: full path to the downloaded video file
    """
    try:
        yt = pytube.YouTube(video_address)
    except Exception:
        sys.exit('Could not find the video!')
    title = yt.title
    streams = yt.streams.filter(only_video=True).all()
    full_name = os.path.join(save_dir, title + '.' + fmt)
    for i in streams:
        if i.mime_type.endswith(fmt) and i.resolution == "720p":
            if os.path.exists(full_name):
                print('Video already exists.')
                return full_name
            print('Downloading {} from {} to {}'.format(
                title, video_address, save_dir))
            i.download(save_dir)
    print('Video saved to {}'.format(video_address))
    return full_name

# example
# if __name__ == '__main__':
#     video_dir = get_video_from_youtube(
#         'https://www.youtube.com/watch?v=XVO9CS8D4hQ', save_dir='./')
#     vidreader = VideoReader(video_dir)
#     vidreader.extractPiece(120, 288, name='cat.mp4')
