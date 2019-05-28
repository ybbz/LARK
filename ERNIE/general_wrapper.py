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
"""General wrapper of Paddle Fluid."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import multiprocessing

import paddle.fluid as fluid

import reader.task_reader as task_reader
import finetune.classifier as classifier
import finetune.sequence_label as sequence_label
from optimization import optimization
from utils.args import print_arguments
from utils.init import init_pretraining_params, init_checkpoint
from finetune_args import parser
import json
import copy
import numpy as np

def global_init(global_args):
    if global_args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    startup_prog = fluid.Program()
    if global_args.random_seed is not None:
        startup_prog.random_seed = global_args.random_seed

    if global_args.do_train:
        exec_strategy = fluid.ExecutionStrategy()
        if global_args.use_fast_executor:
            exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = dev_count
        exec_strategy.num_iteration_per_drop_scope = global_args.num_iteration_per_drop_scope

    exe = fluid.Executor(place)

    global_resource = {}
    global_resource["place"] = place
    global_resource["dev_count"] = dev_count
    global_resource["startup_prog"] = startup_prog
    global_resource["exec_strategy"] = exec_strategy
    global_resource["total_train_steps"] = 0
    global_resource["exe"] = exe

    return global_resource

def task_resource_init(global_resource, args, create_model, task_type):
    place = global_resource["place"]
    dev_count = global_resource["dev_count"]
    startup_prog = global_resource["startup_prog"]

    if task_type == "classifier":
        reader = task_reader.ClassifyReader(
            vocab_path=args.vocab_path,
            label_map_config=args.label_map_config,
            max_seq_len=args.max_seq_len,
            do_lower_case=args.do_lower_case,
            in_tokens=args.in_tokens,
            random_seed=args.random_seed)
        evaluate = classifier.evaluate

    elif task_type == "sequence_label":
        reader = task_reader.SequenceLabelReader(
            vocab_path=args.vocab_path,
            label_map_config=args.label_map_config,
            max_seq_len=args.max_seq_len,
            do_lower_case=args.do_lower_case,
            in_tokens=args.in_tokens,
            random_seed=args.random_seed)
        evaluate = sequence_label.evaluate

    if not (args.do_train or args.do_val or args.do_test):
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")    
    if args.do_train:
        train_data_generator = reader.data_generator(
            input_file=args.train_set,
            batch_size=args.batch_size,
            epoch=args.epoch,
            shuffle=True,
            phase="train")

        num_train_examples = reader.get_num_examples(args.train_set)

        if args.in_tokens:
            max_train_steps = args.epoch * num_train_examples // (
                args.batch_size // args.max_seq_len) // dev_count
        else:
            max_train_steps = args.epoch * num_train_examples // \
                args.batch_size // dev_count

        global_resource["total_train_steps"] += max_train_steps
        warmup_steps = int(max_train_steps * args.warmup_proportion)
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, graph_vars = create_model(
                    args, pyreader_name='train_reader')
                scheduled_lr = optimization(
                    loss=graph_vars["loss"],
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
                    use_fp16=args.use_fp16,
                    loss_scaling=args.loss_scaling)
                fluid.memory_optimize(
                    input_program=train_program,
                    skip_opt_set=[
                        graph_vars["loss"].name,
                        graph_vars["probs"].name,
                    ])

        train_pyreader.decorate_tensor_provider(train_data_generator)

        if warmup_steps > 0:
            graph_vars["learning_rate"] = scheduled_lr


    if args.do_val or args.do_test:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, graph_vars = create_model(
                    args, pyreader_name='test_reader')

        test_prog = test_prog.clone(for_test=True)

    task = {"args": args,
            "evaluate": evaluate,
            "reader": reader,
            "num_train_examples": num_train_examples,
            "max_train_steps": max_train_steps,
            "warmup_steps": warmup_steps,
            "train_program": train_program,
            "train_pyreader": train_pyreader,
            "graph_vars": graph_vars,
            "train_exe": None,
            "test_prog": test_prog,
            "test_pyreader": test_pyreader,
            "test_exe": None}

    return task


def params_init(global_args, global_resource):
    exe = global_resource["exe"]
    startup_prog = global_resource["startup_prog"]
    exe.run(startup_prog)


def task_train_init(global_resource, task):
    place = global_resource["place"]
    exe = global_resource["exe"]
    exec_strategy = global_resource["exec_strategy"]
    args = task["args"]
    
    task["train_exe"] = fluid.ParallelExecutor(
        use_cuda=args.use_cuda,
        loss_name=task["graph_vars"]["loss"].name,
        exec_strategy=exec_strategy,
        main_program=task["train_program"])

    task["train_pyreader"].start()
    task["steps"] = 0
    task["finish"] = False


def run_train(global_resource, task):
    dev_count = global_resource["dev_count"]
    steps = task["steps"]
    args = task["args"]
    train_exe = task["train_exe"]
    reader = task["reader"]
    evaluate = task["evaluate"]
    train_program = task["train_program"]
    train_pyreader = task["train_pyreader"]
    graph_vars = task["graph_vars"]
    warmup_steps = task["warmup_steps"]

    if steps % args.skip_steps != 0:
        train_exe.run(fetch_list=[])
        outputs = None
    else:
        outputs = evaluate(train_exe, train_program, train_pyreader, graph_vars,
                "train", args.num_labels, dev_count)
        if args.verbose:
            verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size()
            verbose += "learning rate: %f" % (outputs["learning_rate"] \
                                             if warmup_steps > 0 else args.learning_rate)
            print(verbose)    

    return outputs


def run_eval(global_resource, task):
    exe = global_resource["exe"]
    steps = task["steps"]
    args = task["args"]
    reader = task["reader"]
    evaluate = task["evaluate"]
    graph_vars = task["graph_vars"]
    test_prog = task["test_prog"]
    test_pyreader = task["test_pyreader"]
    dev_count = global_resource["dev_count"]
    
    if args.do_val:
        test_pyreader.decorate_tensor_provider(
                reader.data_generator(
                        args.dev_set,
                        batch_size=args.batch_size,
                        epoch=1,
                        shuffle=False))
        evaluate(exe, test_prog, test_pyreader, graph_vars,
                "dev", args.num_labels, dev_count)
    
    if args.do_test:
        test_pyreader.decorate_tensor_provider(
                reader.data_generator(
                        args.test_set,
                        batch_size=args.batch_size,
                        epoch=1,
                        shuffle=False))
        evaluate(exe, test_prog, test_pyreader, graph_vars,
                "test", args.num_labels, dev_count)
    
def save_checkpoint(global_resource, task):
    exe = global_resource["exe"]
    steps = task["steps"]
    args = task["args"]
    train_program = task["train_program"]

    save_path = os.path.join(args.checkpoints, "step_" + str(steps))
    fluid.io.save_persistables(exe, save_path, train_program)

