from __future__ import absolute_import

from . import BaseTracker


class IdentityTracker(BaseTracker):

    def __init__(self):
        super(IdentityTracker, self).__init__(
            name='IdentityTracker',
            is_deterministic=True)

    def init(self, image, box):
        self.box = box

    def update(self, image):
        return self.box
