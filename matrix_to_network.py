import warnings
from collections import namedtuple
from functools import partial
import copy
from typing import Optional, Tuple, List, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import networkx as nx
import numpy as np


INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

def is_upper_triangular(matrix):
  """True if matrix is 0 on diagonal and below."""
  for src in range(np.shape(matrix)[0]):
    for dst in range(0, src + 1):
      if matrix[src, dst] != 0:
        return False

  return True

class ModelSpec(object):
  """Model specification given adjacency matrix and labeling."""

  def __init__(self, matrix, ops, data_format='channels_last'):
    """Initialize the module spec.
    Args:
      matrix: ndarray or nested list with shape [V, V] for the adjacency matrix.
      ops: V-length list of labels for the base ops used. The first and last
        elements are ignored because they are the input and output vertices
        which have no operations. The elements are retained to keep consistent
        indexing.
      data_format: channels_last or channels_first.
    Raises:
      ValueError: invalid matrix or ops
    """
    if not isinstance(matrix, np.ndarray):
      matrix = np.array(matrix)
    shape = np.shape(matrix)
    if len(shape) != 2 or shape[0] != shape[1]:
      raise ValueError('matrix must be square')
    if shape[0] != len(ops):
      raise ValueError('length of ops must match matrix dimensions')
    if not is_upper_triangular(matrix):
      raise ValueError('matrix must be upper triangular')

    # Both the original and pruned matrices are deep copies of the matrix and
    # ops so any changes to those after initialization are not recognized by the
    # spec.
    self.original_matrix = copy.deepcopy(matrix)
    self.original_ops = copy.deepcopy(ops)

    self.matrix = copy.deepcopy(matrix)
    self.ops = copy.deepcopy(ops)
    self.valid_spec = True
    self._prune()

    self.data_format = data_format

  def _prune(self):
    """Prune the extraneous parts of the graph.
    General procedure:
      1) Remove parts of graph not connected to input.
      2) Remove parts of graph not connected to output.
      3) Reorder the vertices so that they are consecutive after steps 1 and 2.
    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """
    num_vertices = np.shape(self.original_matrix)[0]

    # DFS forward from input
    visited_from_input = set([0])
    frontier = [0]
    while frontier:
      top = frontier.pop()
      for v in range(top + 1, num_vertices):
        if self.original_matrix[top, v] and v not in visited_from_input:
          visited_from_input.add(v)
          frontier.append(v)

    # DFS backward from output
    visited_from_output = set([num_vertices - 1])
    frontier = [num_vertices - 1]
    while frontier:
      top = frontier.pop()
      for v in range(0, top):
        if self.original_matrix[v, top] and v not in visited_from_output:
          visited_from_output.add(v)
          frontier.append(v)

    # Any vertex that isn't connected to both input and output is extraneous to
    # the computation graph.
    extraneous = set(range(num_vertices)).difference(
        visited_from_input.intersection(visited_from_output))

    # If the non-extraneous graph is less than 2 vertices, the input is not
    # connected to the output and the spec is invalid.
    if len(extraneous) > num_vertices - 2:
      self.matrix = None
      self.ops = None
      self.valid_spec = False
      return

    self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
    self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
    for index in sorted(extraneous, reverse=True):
      del self.ops[index]


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Cell(nn.Module):
    def __init__(self, in_channels, matrix, ops):
        super(Cell, self).__init__()
        # The input ops and the corresponding adj matrix may not necessarily
        # be a 7-node graph. In that case we need to prune the graph to set in order.
        # This is handled by the class ModelSpec.
        # For more on ModelSpec: https://github.com/google-research/nasbench/blob/b94247037ee470418a3e56dcb83814e9be83f3a8/nasbench/lib/model_spec.py#L37

        self.spec = ModelSpec(matrix, ops)
        self.matrix = self.spec.matrix
        self.ops = self.spec.ops
        self.G = nx.from_numpy_matrix(self.matrix, create_using=nx.DiGraph)
        self.num_nodes = len(self.ops)
        
        for i in range(self.num_nodes):
            self.G.nodes[i]['label'] = i
            self.G.nodes[i]['op_label'] = self.ops[i]
            self.G.nodes[i]['incoming'] = [n for n in self.G.reverse().neighbors(i)]
            self.G.nodes[i]['outgoing'] = [n for n in self.G.neighbors(i)]
        
        self.proj_depth = int(in_channels/len(self.G.nodes[self.num_nodes-1]['incoming']))
        self.layers = nn.ModuleList()
        
        for n in range(self.num_nodes):
            node = self.G.nodes[n]
            modules = []
            op_label = node['op_label']
            if 0 in node['incoming']:
                modules.append(BasicConv2d(in_channels, self.proj_depth, kernel_size=1))
            if op_label == CONV1X1 and len(modules) == 0:
                modules.append(BasicConv2d(self.proj_depth, self.proj_depth, kernel_size=1))
            elif op_label == CONV3X3:
                modules.append(BasicConv2d(self.proj_depth, self.proj_depth, kernel_size=3, padding=1))
            elif op_label == MAXPOOL3X3:
                modules.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True))
            if len(modules):
                self.layers.append(nn.Sequential(*modules))

    def forward(self, x):
        self.G.nodes[0]['output'] = x
        for n in range(1,self.num_nodes-1):
            input = 0
            for neighbor in self.G.nodes[n]['incoming']:
                input += self.G.nodes[neighbor]['output']
            self.G.nodes[n]['output'] = self.layers[n-1](input)
        outputs = torch.cat([self.G.nodes[n]['output'] for n in self.G.nodes[6]['incoming']], 1)
        return outputs

class CustomPool(nn.Module):
    def __init__(self, in_channels):
        super(CustomPool, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1x1 = BasicConv2d(in_channels, 2*in_channels, kernel_size=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv1x1(x)
        return x

class CNN(nn.Module):
    def __init__(self, matrix, ops):
        super(CNN, self).__init__()
        self.convstem = BasicConv2d(3, 128, kernel_size=3, padding=1) # 32x32
        
        self.stack1 = nn.ModuleList()
        for _ in range(3):
            self.stack1.append(Cell(128, matrix, ops))
        self.stack1.append(CustomPool(128)) # 16x16

        self.stack2 = nn.ModuleList()
        for _ in range(3):
            self.stack2.append(Cell(256, matrix, ops))
        self.stack2.append(CustomPool(256)) # 8X8

        self.stack3 = nn.ModuleList()
        for _ in range(3):
            self.stack3.append(Cell(512, matrix, ops))
        self.global_pool = nn.MaxPool2d(8)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self,x):
        x = self.convstem(x)
        for i in range(4):
            x = self.stack1[i](x)
        for i in range(4):
            x = self.stack2[i](x)
        for i in range(3):
            x = self.stack3[i](x)
        x = self.global_pool(x).flatten(start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
