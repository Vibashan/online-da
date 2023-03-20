# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn
import copy

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds

import pdb
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.data.detection_utils import convert_image_to_rgb

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage
from collections import defaultdict

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            # if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
            #     eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
            #     log_every_n_seconds(
            #         logging.INFO,
            #         (
            #             f"Inference done {idx + 1}/{total}. "
            #             f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
            #             f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
            #             f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
            #             f"Total: {total_seconds_per_iter:.4f} s/iter. "
            #             f"ETA={eta}"
            #         ),
            #         n=5,
            #     )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    # logger.info(
    #     "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_time_str, total_time / (total - num_warmup), num_devices
    #     )
    # )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    # logger.info(
    #     "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
    #     )
    # )
    
    ###### results = OrderedDict([('bbox', {'AP': 0.454, 'AP50': 0.454, 'AP75': 0.454})])
    results = evaluator.evaluate()

    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def oshot_inference_on_dataset(
    model_teacher, model_student, data_loader, optimizer, cfg, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    start_iter, max_iter = 0, 500#len(data_loader)
    oshot_breakpoints = cfg.OSHOT.OSHOT_BREAKPOINTS

    with EventStorage(start_iter) as storage:

        t_results = [defaultdict(list) for _ in range(len(cfg.OSHOT.OSHOT_BREAKPOINTS))]
        s_results = [defaultdict(list) for _ in range(len(cfg.OSHOT.OSHOT_BREAKPOINTS))]

        t_checkpoint = copy.deepcopy(model_teacher.state_dict())
        s_checkpoint = copy.deepcopy(model_student.state_dict())

        start_data_time = time.perf_counter()
        for data, idx in zip(data_loader, range(start_iter, max_iter)):
            
            model_teacher.load_state_dict(t_checkpoint)
            model_teacher.cuda()
            model_student.load_state_dict(s_checkpoint)
            model_student.cuda()
            
            # with torch.no_grad():
            #     outputs = model(data)
            #     torch.cuda.synchronize()
            #     output = [o.to(cpu_device) for o in outputs]

            for oshot_it in range(cfg.OSHOT.OSHOT_ITERATIONS):
                model_teacher.eval()
                model_student.train()

                optimizer.zero_grad()
                
                with torch.no_grad():
                    _, teacher_features, teacher_proposals, teacher_results = model_teacher(data, mode="train")

                teacher_pseudo_proposals, num_rpn_proposal = process_pseudo_label(teacher_proposals, 0.8, "rpn", "thresholding")
                teacher_pseudo_results, num_roih_proposal = process_pseudo_label(teacher_results, 0.8, "roih", "thresholding")
                teacher_mem_results, num_mem_proposal = process_pseudo_label(teacher_results, 0.8, "roih", "thresholding")

                #visualize_proposals(cfg, data, teacher_proposals, num_rpn_proposal, "rpn", metadata)
                #visualize_proposals(cfg, data, teacher_results, num_roih_proposal, "roih", metadata)

                loss_dict = model_student(data, cfg, model_teacher, teacher_features, teacher_proposals, teacher_pseudo_results, teacher_mem_results, mode="train")
                losses = sum(loss_dict.values())
                #losses = loss_dict["feat_consistency"]
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
                #print("oshot_itr: ", oshot_it, "lr:", optimizer.param_groups[0]["lr"], ''.join(['{0}: {1}, '.format(k, v.item()) for k,v in loss_dict.items()]))

                losses.backward()
                optimizer.step()
                
                if (oshot_it + 1) in oshot_breakpoints:

                    model_teacher.eval()
                    with torch.no_grad():
                        outputs = model_teacher(data)
                    copied_data = copy.deepcopy(data)

                    t_results[oshot_breakpoints.index(oshot_it+1)]["inputs"].append(copied_data)
                    t_results[oshot_breakpoints.index(oshot_it+1)]["outputs"].append(outputs)

                    model_student.eval()
                    with torch.no_grad():
                        outputs = model_student(data)

                    s_results[oshot_breakpoints.index(oshot_it+1)]["inputs"].append(copied_data)
                    s_results[oshot_breakpoints.index(oshot_it+1)]["outputs"].append(outputs)
                    model_student.train()

                # new_teacher_dict = update_teacher_model(model_student, model_teacher, keep_rate=0.9)
                # model_teacher.load_state_dict(new_teacher_dict)
                # model_teacher.eval()
            print(idx, max_iter)
                
            #start_compute_time = time.perf_counter()
            #outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # total_compute_time += time.perf_counter() - start_compute_time

            # start_eval_time = time.perf_counter()
            # evaluator.process(inputs, outputs)
            # total_eval_time += time.perf_counter() - start_eval_time

            # iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            # data_seconds_per_iter = total_data_time / iters_after_start
            # compute_seconds_per_iter = total_compute_time / iters_after_start
            # eval_seconds_per_iter = total_eval_time / iters_after_start
            # total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            # if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
            #     eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
            #     log_every_n_seconds(
            #         logging.INFO,
            #         (
            #             f"Inference done {idx + 1}/{total}. "
            #             f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
            #             f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
            #             f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
            #             f"Total: {total_seconds_per_iter:.4f} s/iter. "
            #             f"ETA={eta}"
            #         ),
            #         n=5,
            #     )
            #start_data_time = time.perf_counter()
    
    t_evaluations = []
    for idx,pred in enumerate(t_results):
        for input, output in zip(pred['inputs'], pred['outputs']):
            evaluator.process(input, output)

        t_evaluations.append(evaluator.evaluate())
    
    s_evaluations = []
    for idx,pred in enumerate(s_results):
        for input, output in zip(pred['inputs'], pred['outputs']):
            evaluator.process(input, output)

        s_evaluations.append(evaluator.evaluate())
        
    return t_evaluations, s_evaluations

    # Measure the time only for this worker (before the synchronization barrier)
    #total_time = time.perf_counter() - start_time
    #total_time_str = str(datetime.timedelta(seconds=total_time))
    # logger.info(
    #     "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_time_str, total_time / (total - num_warmup), num_devices
    #     )
    # )
    #total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    # logger.info(
    #     "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
    #     )
    # )
    
    ###### results = OrderedDict([('bbox', {'AP': 0.454, 'AP50': 0.454, 'AP75': 0.454})])
    # results = evaluator.evaluate()

    # # An evaluator may return None when not in main process.
    # # Replace it by an empty dict instead to make it easier for downstream code to handle
    # if results is None:
    #     results = {}
    # return results

def inference_on_corruption_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            base_dict = {k: v for k, v in inputs[0].items() if "image" not in k}
            base_dict["image_id"] = inputs[0]["image_id"]
            for severity in range(0,5):
                corrupt_inputs = base_dict.copy()
                corrupt_inputs["image"] = inputs[0]["image_" + str(severity)]
                corrupt_inputs = [corrupt_inputs]
                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                outputs = model(corrupt_inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                evaluator.process(corrupt_inputs, outputs)
                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                # if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                #     eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                #     log_every_n_seconds(
                #         logging.INFO,
                #         (
                #             f"Inference done {idx + 1}/{total}. "
                #             f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                #             f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                #             f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                #             f"Total: {total_seconds_per_iter:.4f} s/iter. "
                #             f"ETA={eta}"
                #         ),
                #         n=5,
                #     )
                start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    # logger.info(
    #     "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_time_str, total_time / (total - num_warmup), num_devices
    #     )
    # )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    # logger.info(
    #     "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
    #     )
    # )
    
    ###### results = OrderedDict([('bbox', {'AP': 0.454, 'AP50': 0.454, 'AP75': 0.454})])
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def threshold_bbox(proposal_bbox_inst, thres=0.7, proposal_type="roih"):
    if proposal_type == "rpn":
        valid_map = proposal_bbox_inst.objectness_logits > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
            valid_map
        ]
    elif proposal_type == "roih":
        valid_map = proposal_bbox_inst.scores > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
        new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

    return new_proposal_inst

def process_pseudo_label(proposals_rpn_k, cur_threshold, proposal_type, psedo_label_method=""):
    list_instances = []
    num_proposal_output = 0.0
    for proposal_bbox_inst in proposals_rpn_k:
        # thresholding
        if psedo_label_method == "thresholding":
            proposal_bbox_inst = threshold_bbox(
                proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
            )
        else:
            raise ValueError("Unkown pseudo label boxes methods")
        num_proposal_output += len(proposal_bbox_inst)
        list_instances.append(proposal_bbox_inst)
    num_proposal_output = num_proposal_output / len(proposals_rpn_k)
    return list_instances, num_proposal_output

@torch.no_grad()
def update_teacher_model(model_student, model_teacher, keep_rate=0.996):
    if comm.get_world_size() > 1:
        student_model_dict = {
            key[7:]: value for key, value in model_student.state_dict().items()
        }
    else:
        student_model_dict = model_student.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    return new_teacher_dict
