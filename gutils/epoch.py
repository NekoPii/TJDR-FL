#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:maojingxin
"""


class Epoch(object):
    def __init__(self, cloud_epoch, edge_epoch, client_epoch, num_cloud_ep: int = None, num_edge_ep: int = None, num_client_ep: int = None):
        assert cloud_epoch >= 0
        assert edge_epoch is None or edge_epoch >= 0
        assert client_epoch is None or client_epoch >= 0

        self._cloud_epoch = cloud_epoch
        self._edge_epoch = edge_epoch if edge_epoch is not None else None
        self._client_epoch = client_epoch if client_epoch is not None else None

        self._num_cloud_ep = num_cloud_ep
        self._num_edge_ep = num_edge_ep
        self._num_client_ep = num_client_ep

    def c_to_str(self):
        return "{}".format(self._cloud_epoch)

    def ce_to_str(self):
        assert self._edge_epoch is not None
        return "{}-{}".format(self._cloud_epoch, self._edge_epoch)

    def cec_to_str(self):
        assert self._edge_epoch is not None and self._client_epoch is not None
        return "{}-{}-{}".format(self._cloud_epoch, self._edge_epoch, self._client_epoch)

    def total_client_ep(self, ignore_cloud=False):
        assert self._num_edge_ep > 0 and self._num_client_ep > 0 and self._edge_epoch > 0 and self._client_epoch > 0
        total_num = (self._edge_epoch - 1) * self._num_client_ep + self._client_epoch if ignore_cloud \
            else ((self._cloud_epoch - 1) * self._num_edge_ep + (self._edge_epoch - 1)) * self._num_client_ep + self._client_epoch
        return total_num

    def total_edge_ep(self, ignore_cloud=False):
        assert self._num_edge_ep > 0 and self._edge_epoch >= 0
        total_num = self._edge_epoch if ignore_cloud else (self._cloud_epoch - 1) * self._num_edge_ep + self._edge_epoch
        return total_num

    def total_cloud_ep(self):
        return self._cloud_epoch

    def cloud_epoch_plus(self):
        self._cloud_epoch += 1

    def edge_epoch_plus(self):
        self._edge_epoch += 1

    def client_epoch_plus(self):
        self._client_epoch += 1

    def update(self, new_epoch):
        if new_epoch.cloud_epoch is not None:
            self._cloud_epoch = new_epoch.cloud_epoch
        if new_epoch.edge_epoch is not None:
            self._edge_epoch = new_epoch.edge_epoch
        if new_epoch.client_epoch is not None:
            self._client_epoch = new_epoch.client_epoch
        if new_epoch.num_client_ep is not None:
            self._num_client_ep = new_epoch.num_client_ep
        if new_epoch.num_edge_ep is not None:
            self._num_edge_ep = new_epoch.num_edge_ep
        if new_epoch.num_cloud_ep is not None:
            self._num_cloud_ep = new_epoch.num_cloud_ep

    def reset(self, cloud_epoch=False, edge_epoch=False, client_epoch=False):
        if cloud_epoch:
            self._cloud_epoch = 0
        if edge_epoch:
            self._edge_epoch = 0
        if client_epoch:
            self._client_epoch = 0

    @property
    def client_epoch(self):
        return self._client_epoch

    @property
    def edge_epoch(self):
        return self._edge_epoch

    @property
    def cloud_epoch(self):
        return self._cloud_epoch

    @property
    def num_client_ep(self):
        return self._num_client_ep

    @property
    def num_edge_ep(self):
        return self._num_edge_ep

    @property
    def num_cloud_ep(self):
        return self._num_cloud_ep

    def serialize(self):
        d = self.__dict__
        new_d = dict()
        for k, v in d.items():
            new_d[k[1:]] = v
        return new_d


if __name__ == "__main__":
    a = Epoch(1, 1, None, None, 10, 1)
    b = Epoch(**a.serialize())
    print(a.serialize())
    pass
