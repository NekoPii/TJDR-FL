#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:maojingxin
"""


class ID(object):
    def __init__(self, num_id: int, uuid: str):
        self._num_id = num_id
        self._uuid = uuid
        self._parent_id = None
        self._children_id = []

    def __eq__(self, other):
        return self._num_id == other.nid and self._uuid == other.fid

    def set_parent_id(self, parent_id):
        self._parent_id = parent_id

    def add_child_id(self, child_id):
        self._children_id.append(child_id)

    def remove_child_id(self, child_id):
        if child_id:
            for _child_id in self._children_id:
                if child_id == _child_id:
                    self._children_id.remove(child_id)
            child_id.set_parent_id(None)

    def remove_parent_id(self):
        if self._parent_id:
            self.parent_id.remove_child_id(self)

    @property
    def parent_id(self):
        return self._parent_id

    @property
    def children_id(self):
        return self._children_id

    @property
    def nid(self):
        return self._num_id

    @property
    def fid(self):
        return self._uuid

    @property
    def sid(self):
        return self._uuid[:4]
