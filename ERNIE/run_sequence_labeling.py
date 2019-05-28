#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning on classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import multiprocessing
import commands

import paddle.fluid as fluid

from utils.args import print_arguments
from finetune_args import parser
import json
import copy
import numpy as np
from general_wrapper import *
from finetune.sequence_label import ernie_pyreader
import paddlehub as hub

args = parser.parse_args()

def create_model(args, pyreader_name, is_prediction=False):
    pyreader, ernie_inputs, labels = ernie_pyreader(pyreader_name, args.max_seq_len)
    module = hub.Module(name="ernie")
    inputs, outputs, program = module.context(trainable="True", max_seq_len=args.max_seq_len)
    input_dict = {
        inputs["input_ids"].name: ernie_inputs["src_ids"],
        inputs["position_ids"].name: ernie_inputs["pos_ids"],
        inputs["segment_ids"].name: ernie_inputs["sent_ids"],
        inputs["input_mask"].name: ernie_inputs["input_mask"],
    }

    hub.connect_program(
        pre_program=fluid.default_main_program(),
        next_program=program,
        input_dict=input_dict)

    enc_out = fluid.layers.dropout(
        x=outputs["sequence_output"],
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")
    logits = fluid.layers.fc(
        enc_out,
        size=args.num_labels,
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(
            name="cls_seq_label_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_seq_label_out_b",
            initializer=fluid.initializer.Constant(0.)))

    ret_labels = fluid.layers.reshape(x=labels, shape=[-1, 1])
    ret_infers = fluid.layers.reshape(
        x=fluid.layers.argmax(
            logits, axis=2), shape=[-1, 1])

    labels = fluid.layers.flatten(labels, axis=2)
    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=fluid.layers.flatten(
            logits, axis=2),
        label=labels,
        return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)

    if args.use_fp16 and args.loss_scaling > 1.0:
        loss *= args.loss_scaling

    graph_vars = {
        "loss": loss,
        "probs": probs,
        "labels": ret_labels,
        "infers": ret_infers,
        "seq_lens": ernie_inputs["seq_lens"]
    }
    
    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars

def main(args):
    global_resource = global_init(args)
    task = task_resource_init(global_resource, args, create_model, task_type="sequence_label")
    params_init(args, global_resource)

    if args.do_train:
        task_train_init(global_resource, task)
        time_begin = time.time()
        while True:
            steps = task["steps"]
            args = task["args"]
            reader = task["reader"]
            try:
                steps += 1
                task["steps"] = steps
                outputs = run_train(global_resource, task)

                if steps % args.skip_steps == 0:
                    current_example, current_epoch = reader.get_train_progress()
                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                          "f1: %f, precision: %f, recall: %f, speed: %f steps/s"
                           % (current_epoch, current_example, task["num_train_examples"],
                             steps, outputs["loss"], outputs["f1"],
                             outputs["precision"], outputs["recall"],
                             args.skip_steps / used_time))

                    time_begin = time.time()

                if steps % args.save_steps == 0:
                    save_checkpoint(global_resource, task)
                
                if steps % args.validation_steps == 0:
                    run_eval(global_resource, task)

            except fluid.core.EOFException:
                save_checkpoint(global_resource, task)
                task["train_pyreader"].reset()
                break

    print("\n=======Final evaluation ========")
    run_eval(global_resource, task)

if __name__ == '__main__':
    print_arguments(args)
    main(args)

