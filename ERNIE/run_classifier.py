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

import paddle.fluid as fluid

import reader.task_reader as task_reader
from finetune.classifier import ernie_pyreader, evaluate
from optimization import optimization
from utils.args import print_arguments
from utils.init import init_pretraining_params, init_checkpoint
from finetune_args import parser
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

    cls_feats = fluid.layers.dropout(
        x=outputs["pooled_output"],
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")
    logits = fluid.layers.fc(
        input=cls_feats,
        size=args.num_labels,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))

    if is_prediction:
        probs = fluid.layers.softmax(logits)
        feed_targets_name = [
            src_ids.name, pos_ids.name, sent_ids.name, input_mask.name
        ]
        return pyreader, probs, feed_targets_name

    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)

    if args.use_fp16 and args.loss_scaling > 1.0:
        loss *= args.loss_scaling

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)

    graph_vars = {
        "loss": loss,
        "probs": probs,
        "accuracy": accuracy,
        "labels": labels,
        "num_seqs": num_seqs,
        "qids": ernie_inputs["qids"]
    }
    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars


def main(args):
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    reader = task_reader.ClassifyReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=args.in_tokens,
        random_seed=args.random_seed)

    if not (args.do_train or args.do_val or args.do_test):
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

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
            max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

        warmup_steps = int(max_train_steps * args.warmup_proportion)
        print("Device count: %d" % dev_count)
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)
        print("Num warmup steps: %d" % warmup_steps)

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
                        graph_vars["accuracy"].name,
                        graph_vars["num_seqs"].name,
                    ])

        if args.verbose:
            if args.in_tokens:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program,
                    batch_size=args.batch_size // args.max_seq_len)
            else:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program, batch_size=args.batch_size)
            print("Theoretical memory usage in training: %.3f - %.3f %s" %
                  (lower_mem, upper_mem, unit))

    if args.do_val or args.do_test:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, graph_vars = create_model(
                    args, pyreader_name='test_reader')

        test_prog = test_prog.clone(for_test=True)

    exe.run(startup_prog)

    if args.do_train:
        exec_strategy = fluid.ExecutionStrategy()
        if args.use_fast_executor:
            exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = dev_count
        exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope

        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=graph_vars["loss"].name,
            exec_strategy=exec_strategy,
            main_program=train_program)

        train_pyreader.decorate_tensor_provider(train_data_generator)
    else:
        train_exe = None

    if args.do_train:
        train_pyreader.start()
        steps = 0
        if warmup_steps > 0:
            graph_vars["learning_rate"] = scheduled_lr

        time_begin = time.time()
        while True:
            try:
                steps += 1
                if steps % args.skip_steps != 0:
                    train_exe.run(fetch_list=[])
                else:
                    outputs = evaluate(train_exe, train_program, train_pyreader,
                                       graph_vars, "train")

                    if args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size(
                        )
                        verbose += "learning rate: %f" % (
                            outputs["learning_rate"]
                            if warmup_steps > 0 else args.learning_rate)
                        print(verbose)

                    current_example, current_epoch = reader.get_train_progress()
                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                          "ave acc: %f, speed: %f steps/s" %
                          (current_epoch, current_example, num_train_examples,
                           steps, outputs["loss"], outputs["accuracy"],
                           args.skip_steps / used_time))
                    time_begin = time.time()

                if steps % args.save_steps == 0:
                    save_path = os.path.join(args.checkpoints,
                                             "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)

                if steps % args.validation_steps == 0:
                    # evaluate dev set
                    if args.do_val:
                        test_pyreader.decorate_tensor_provider(
                            reader.data_generator(
                                args.dev_set,
                                batch_size=args.batch_size,
                                epoch=1,
                                shuffle=False))
                        evaluate(exe, test_prog, test_pyreader, graph_vars,
                                 "dev")
                    # evaluate test set
                    if args.do_test:
                        test_pyreader.decorate_tensor_provider(
                            reader.data_generator(
                                args.test_set,
                                batch_size=args.batch_size,
                                epoch=1,
                                shuffle=False))
                        evaluate(exe, test_prog, test_pyreader, graph_vars,
                                 "test")
            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

    # final eval on dev set
    if args.do_val:
        test_pyreader.decorate_tensor_provider(
            reader.data_generator(
                args.dev_set,
                batch_size=args.batch_size,
                epoch=1,
                shuffle=False))
        print("Final validation result:")
        evaluate(exe, test_prog, test_pyreader, graph_vars, "dev")

    # final eval on test set
    if args.do_test:
        test_pyreader.decorate_tensor_provider(
            reader.data_generator(
                args.test_set,
                batch_size=args.batch_size,
                epoch=1,
                shuffle=False))
        print("Final test result:")
        evaluate(exe, test_prog, test_pyreader, graph_vars, "test")


if __name__ == '__main__':
    print_arguments(args)
    main(args)
