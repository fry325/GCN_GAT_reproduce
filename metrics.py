#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 17:16:28 2021

@author: Gong Dongsheng
"""

import torch


def masked_softmax_cross_entropy(org_loss_func, preds, labels, mask):
    loss = org_loss_func(preds, labels)
    _mask = torch.FloatTensor(mask)
    _mask /= _mask.sum()
    loss *= _mask
    return loss.mean()


def masked_accuracy(preds, labels, mask):
    acc = torch.eq(preds.argmax(1), labels).float()
    _mask = torch.FloatTensor(mask)
    _mask /= _mask.sum()
    acc *= _mask
    return acc.sum().item()