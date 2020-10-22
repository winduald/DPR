import argparse
import glob
import logging
import math
import os
import random
import time
from collections import OrderedDict


import torch

from typing import Tuple
from torch import nn
from torch import Tensor as T
import copy

from dpr.models import init_triangle_encoder_components
from dpr.models.biencoder import BiEncoder
from dpr.options import add_encoder_params, add_training_params, setup_args_gpu, set_seed, print_args, \
    get_encoder_params_state, add_tokenizer_params, set_encoder_params_from_state
from dpr.utils.data_utils import ShardedDataIterator
# from dpr.utils.dist_utils import all_gather_list
# from dpr.utils.model_utils import setup_for_distributed_mode, move_to_device, get_schedule_linear, CheckpointState, \
#     get_model_file, get_model_obj, load_states_from_checkpoint
from dpr.utils.model_utils import get_model_file, load_states_from_checkpoint, \
                                setup_for_distributed_mode, get_schedule_linear

from train_dense_encoder import BiEncoderTrainer, logger, _do_biencoder_fwd_pass

class TriangleEncoder(nn.Module):

    def __init__(self, qqmodel: nn.Module, qamodel: nn.Module, aamodel: nn.Module):
        super(TriangleEncoder, self).__init__()
        self.qqbiencoder = qqmodel
        self.qabiencoder = qamodel
        self.aabiencoder = aamodel



class BiEncoderTrainerMultiTask(BiEncoderTrainer):

    def __init__(self, args):
        self.args = args
        self.shard_id = args.local_rank if args.local_rank != -1 else 0
        self.distributed_factor = args.distributed_world_size or 1

        logger.info("***** Initializing components for training *****")

        # if model file is specified, encoder parameters from saved state should be used for initialization
        model_file = get_model_file(self.args, self.args.checkpoint_file_name)
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            # saved_state = self.remapping_saved_model_dict(saved_state, args)
            set_encoder_params_from_state(saved_state.encoder_params, args)

        tensorizer, qqmodel, qamodel, aamodel, optimizer = init_triangle_encoder_components(args.encoder_model_type, args)

        triangle_encoder = TriangleEncoder(qqmodel, qamodel, aamodel)
        #currently diable distributed training ????
        # triangle_encoder, optimizer = setup_for_distributed_mode(triangle_encoder, optimizer, args.device, args.n_gpu,
        #                                               args.local_rank,
        #                                               args.fp16,
        #                                              args.fp16_opt_level)
        triangle_encoder.to(args.device)
        self.triangle_encoder = triangle_encoder
        # self.qqbiencoder = qqmodel
        # self.qabiencoder = qamodel
        # self.aabiencoder = aamodel
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        if saved_state:
            self.biencoder = self.triangle_encoder.qabiencoder 
            self._load_saved_state(saved_state)
            self.biencoder = None 


    def run_train(self, eval_task_name):
        '''
        This function supports three tasks: question similarity, quesiton answering, answer similarity
        args will have the argument to enble these tasks
        '''

        #hack here

        args = self.args
        upsample_rates = None
        if args.train_files_upsample_rates is not None:
            upsample_rates = eval(args.train_files_upsample_rates)

        qqs_train_iterator, aas_train_iterator, qa_train_iterator = None, None, None
        #create learning schedule
        self.multi_task_schedule = {'qqs': args.num_qqs_data_repetition, 
                                    'aas': args.num_aas_data_repetition, 
                                    'qa': args.num_qa_data_repetition} #askubuntu
        max_iterations = 0
        if args.question_sim_train_file:
            qqs_train_iterator = self.get_data_iterator(args.question_sim_train_file, args.batch_size,
                                                    shuffle=True,
                                                    shuffle_seed=args.seed, offset=self.start_batch,
                                                    upsample_rates=upsample_rates)
            logger.info("qqs: Total iterations per epoch=%d", qqs_train_iterator.max_iterations)
            max_iterations += self.multi_task_schedule['qqs'] * qqs_train_iterator.max_iterations
        else:
            self.multi_task_schedule['qqs'] = 0
        
        if args.answer_sim_train_file:
            aas_train_iterator = self.get_data_iterator(args.answer_sim_train_file, args.batch_size,
                                                    shuffle=True,
                                                    shuffle_seed=args.seed, offset=self.start_batch,
                                                    upsample_rates=upsample_rates)
            logger.info("aas: Total iterations per epoch=%d", aas_train_iterator.max_iterations)
            max_iterations += self.multi_task_schedule['aas'] * aas_train_iterator.max_iterations
        else:
            self.multi_task_schedule['aas'] = 0


        if args.qa_train_file:
            qa_train_iterator = self.get_data_iterator(args.qa_train_file, args.batch_size,
                                                    shuffle=True,
                                                    shuffle_seed=args.seed, offset=self.start_batch,
                                                    upsample_rates=upsample_rates)
            logger.info("qa: Total iterations per epoch=%d", qa_train_iterator.max_iterations)
            max_iterations += self.multi_task_schedule['qa'] * qa_train_iterator.max_iterations
        else:
            self.multi_task_schedule['qa'] = 0

        
        # if qqs_train_iterator is not None:
        #     max_iterations += self.multi_task_schedule['qqs'] * qqs_train_iterator.max_iterations
        # if aas_train_iterator is not None:
        #     max_iterations += self.multi_task_schedule['aas'] * aas_train_iterator.max_iterations
        # if qa_train_iterator is not None:
        #     max_iterations += self.multi_task_schedule['qa'] * qa_train_iterator.max_iterations
        logger.info("estimated number of iterations is %d", max_iterations)

        # if args.num_iterations_per_epoch < 0:    
        #     if args.question_sim_train_file:
        #         max_iterations = qqs_train_iterator.max_iterations
        #     elif args.qa_train_file:
        #         max_iterations = qa_train_iterator.max_iterations
        #     else:
        #         max_iterations = aas_train_iterator.max_iterations
        # else:
        #     max_iterations = args.num_iterations_per_epoch
        # logger.info("max_iterations %d" %max_iterations)

        updates_per_epoch = max_iterations // args.gradient_accumulation_steps
        total_updates = max(updates_per_epoch * (args.num_train_epochs - self.start_epoch - 1), 0) + \
                        (max_iterations - self.start_batch) // args.gradient_accumulation_steps
        logger.info(" Total updates=%d", total_updates)
        warmup_steps = args.warmup_steps
        scheduler = get_schedule_linear(self.optimizer, warmup_steps, total_updates)
        
        if self.scheduler_state:
            logger.info("Loading scheduler state %s", self.scheduler_state)
            scheduler.load_state_dict(self.scheduler_state)

        eval_step = math.ceil(updates_per_epoch / args.eval_per_epoch)
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        for epoch in range(self.start_epoch, int(args.num_train_epochs)):
            logger.info("***** Epoch %d *****", epoch)
            self._train_epoch(scheduler, epoch, eval_step, qqs_train_iterator, \
                            aas_train_iterator, qa_train_iterator, max_iterations, eval_task_name)

        if args.local_rank in [-1, 0]:
            logger.info('Training finished. Best validation checkpoint %s', self.best_cp_name)
            logger.info('Best validation results ', str(self.best_validation_result))

    def get_batch_from_multi_tasks(self, epoch,
                    qqs_train_data_iterator: ShardedDataIterator,
                     aas_train_data_iterator: ShardedDataIterator,
                     qa_train_data_iterator: ShardedDataIterator,):
        # it = 0
        # qqs_batch_gen = qqs_train_data_iterator.iterate_data_rotate(epoch=epoch) if qqs_train_data_iterator else None
        # aas_batch_gen = aas_train_data_iterator.iterate_data_rotate(epoch=epoch) if aas_train_data_iterator else None
        # qa_batch_gen = qa_train_data_iterator.iterate_data_rotate(epoch=epoch) if qa_train_data_iterator else None
        # while True:
        #     it += 1
        #     if qqs_batch_gen:
        #         for j in range(self.multi_task_scheduler['qqs']):
        #             yield next(qqs_batch_gen), 'qqs', qqs_train_data_iterator
        #     if aas_batch_gen:
        #         for j in range(self.multi_task_scheduler['aas']):
        #             yield next(self.aas_batch_gen), 'aas', aas_train_data_iterator
        #     if qa_batch_gen:
        #         for j in range(self.multi_task_scheduler['qa']):
        #             yield next(qa_batch_gen), 'qa', qa_train_data_iterator

        # it = 0
        scheduling = copy.deepcopy(self.multi_task_schedule)
        generator_map = {'qqs': qqs_train_data_iterator.iterate_data(epoch=epoch) if qqs_train_data_iterator else None, 
                        'aas': aas_train_data_iterator.iterate_data(epoch=epoch) if aas_train_data_iterator else None, 
                        'qa': qa_train_data_iterator.iterate_data(epoch=epoch) if qa_train_data_iterator else None}
        iterator_map = {'qqs': qqs_train_data_iterator, 'aas': aas_train_data_iterator, 'qa': qa_train_data_iterator}

        def get_next_item_for(task_name):
            # a task reach to the end of generator and it is the last round, just return None.
            # if reaching the endo of generator but not the last round, recreate a new generator
            if scheduling[task_name] > 0:
                next_item = next(generator_map[task_name], None)
                if next_item is None:
                    scheduling[task_name] -= 1 # restart
                    if scheduling[task_name] > 0: # otherwise, reach to the last round
                        generator_map[task_name] = iterator_map[task_name].iterate_data(epoch=epoch)
                        next_item = next(generator_map[task_name], None)
                        assert next_item is not None, "no data exists in this generator"
                        return next_item, task_name, iterator_map[task_name]
                    else:
                        return None
                else:
                    return next_item, task_name, iterator_map[task_name]
            else:
                return None
        all_queue_empty = False
        it = 0        
        # breakpoint()
        while all_queue_empty != True:
            all_queue_empty = True # if all queue return None, the loop ends
            for tn in ['qqs', 'aas', 'qa']:
                next_item = get_next_item_for(tn)
                if next_item is not None:
                    all_queue_empty = False
                    # if it % 100 == 0:
                    #     print(it)
                    # breakpoint()
                    it += 1
                    yield next_item
            
            

            

        
    # def zero_grad():
    #     #make grad to be zero
    #     #qa biencode contains all parameters
    #     self.qabiencoder.zero_grad()


    def _train_epoch(self, scheduler, epoch: int, eval_step: int,
                     qqs_train_data_iterator: ShardedDataIterator,
                     aas_train_data_iterator: ShardedDataIterator,
                     qa_train_data_iterator: ShardedDataIterator, 
                     max_iterations: int, # this parameter will be re computed
                     eval_task_name: str):
        # breakpoint()
        args = self.args
        rolling_train_loss = 0.0
        log_result_step = args.log_batch_step
        rolling_loss_step = args.train_rolling_loss_step
        num_hard_negatives = args.hard_negatives
        num_other_negatives = args.other_negatives
        seed = args.seed

        # multi_task_schedule = {'qqs': 1, 'aas': 1, 'qa': 20} # semeval
        # multi_task_schedule = {'qqs': 1, 'aas': 1, 'qa': args.num_qa_examples_per_qqs_example} #askubuntu

        # self.qqbiencoder.train()
        # self.aabiencoder.train()
        # self.qabiencoder.train()
        self.triangle_encoder.train()

        # self.biencoder.train()
        # to detmine the number of iterations for epoch

        epoch_batches = max_iterations
        data_iteration = 0
        batch_gen = self.get_batch_from_multi_tasks(epoch, 
                                                    qqs_train_data_iterator,
                                                    aas_train_data_iterator,
                                                    qa_train_data_iterator)
        # num_iteration_per_epoch = max_iterations #this may have isssue ?????
        #we will track three lossexs
        # qq_epoch_loss, aa_epoch_loss, qa_epoch_loss = 0, 0, 0
        # qq_epoch_correct_predictions, aa_epoch_correct_predictions, qa_epoch_correct_predictions = 0, 0, 0
        # qq_rolling_train_loss, aa_rolling_train_loss, qa_rolling_train_loss = 0.0, 0.0, 0.0
        # qq_i, aa_i, qa_i = [0], [0], [0]
        epoch_losses = {'qqs': 0.0, 'aas': 0.0, 'qa': 0.0}
        epoch_correct_predictions = {'qqs': 0, 'aas': 0, 'qa': 0}
        rolling_train_loss = {'qqs': 0.0, 'aas': 0.0, 'qa': 0.0}
        iterations = {'qqs': 0, 'aas': 0, 'qa': 0}
        cur_biencoder = {'qqs': self.triangle_encoder.qqbiencoder, \
                        'aas': self.triangle_encoder.aabiencoder, \
                        'qa': self.triangle_encoder.qabiencoder}

        # breakpoint()
        # for i in range(num_iteration_per_epoch): # deprecate number of iterations
        it = 0
        next_batch = next(batch_gen, None)
        while next_batch is not None:
            samples_batch, task_name, train_data_iterator = next_batch
            # next_batch = next(batch_gen, None)
            # continue
            # samples_batch, task_name, train_data_iterator = next(batch_gen)
            # for i, samples_batch in enumerate(train_data_iterator.iterate_data_rotate(epoch=epoch)):
            # to be able to resume shuffled ctx- pools
            # data_iteration = train_data_iterator.get_iteration()
            # random.seed(seed + epoch + data_iteration)

            data_iteration = train_data_iterator.get_iteration()
            random.seed(seed + epoch + data_iteration)
            biencoder_batch = BiEncoder.create_biencoder_input(samples_batch, self.tensorizer,
                                                               args.use_title_in_ctx,
                                                               num_hard_negatives, num_other_negatives, shuffle=True,
                                                               shuffle_positives=args.shuffle_positive_ctx
                                                               )
            loss, correct_cnt = _do_biencoder_fwd_pass(cur_biencoder[task_name], biencoder_batch, self.tensorizer, args)
            epoch_correct_predictions[task_name] += correct_cnt
            epoch_losses[task_name] += loss.item()
            rolling_train_loss[task_name] += loss.item()
            # print(loss.item())

            if args.fp16:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), args.max_grad_norm)
            else:
                loss.backward()
                if args.max_grad_norm > 0:
                    #clip biencoder
                    torch.nn.utils.clip_grad_norm_(self.triangle_encoder.parameters(), args.max_grad_norm)

            if (it + 1) % args.gradient_accumulation_steps == 0:
                #this update will consider all tasks. So, each task will accumulate gradients.
                #here all accumulated gradients will be updated
                self.optimizer.step()
                scheduler.step()
                self.triangle_encoder.zero_grad()

            if it % log_result_step == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    'Epoch: %d: Step: %d/%d, task=%s, loss=%f, lr=%f', epoch, it, epoch_batches, task_name ,loss.item(), lr)

            #report average loss of all tasks
            if (iterations[task_name] + 1) % rolling_loss_step == 0:
            # if (i + 1) % rolling_loss_step == 0:
                logger.info('Train batch %d', data_iteration)
                latest_rolling_train_av_loss = rolling_train_loss[task_name] / rolling_loss_step
                logger.info('Task: %s  Avg. loss per last %d batches: %f', task_name, rolling_loss_step, latest_rolling_train_av_loss)
                rolling_train_loss[task_name] = 0.0

            #no need to dev?
            # if (it + 1) % eval_step == 0:
            #     logger.info('Validation: Epoch: %d Step: %d/%d', epoch, data_iteration, epoch_batches)
            #     self.validate_and_save(epoch, train_data_iterator.get_iteration(), scheduler)
            #     biencoder.train()
            #iteration add 1
            # breakpoint()
            iterations[task_name] += 1
            it += 1
            next_batch = next(batch_gen, None)

        self.validate_and_save_by_task(epoch, data_iteration, eval_task_name, scheduler)
        # breakpoint()

        # epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        qq_epoch_loss = (epoch_losses['qqs'] / iterations['qqs']) if iterations['qqs'] > 0 else 0
        aa_epoch_loss = (epoch_losses['aas'] / iterations['aas']) if iterations['aas'] > 0 else 0
        qa_epoch_loss = (epoch_losses['qa'] / iterations['qa']) if iterations['qa'] > 0 else 0
        logger.info('Av Loss per epoch by tasks.  qqs %f, aas %f, qa %f', \
            qq_epoch_loss, aa_epoch_loss, qa_epoch_loss)
        logger.info('epoch total correct predictions. qqs %d, aas %d, qa %d', \
            epoch_correct_predictions['qqs'], epoch_correct_predictions['aas'], epoch_correct_predictions['qa'])

    def set_biencoder_and_dev_file_by_task(self, task_name):
        if task_name == 'qqs':
            self.biencoder = self.triangle_encoder.qqbiencoder
            self.args.dev_file = self.args.question_sim_dev_file
        elif task_name == 'aas':
            self.biencoder = self.triangle_encoder.aabiencoder
            self.args.dev_file = self.args.answer_sim_dev_file
        elif task_name == 'qa':
            self.biencoder = self.triangle_encoder.qabiencoder
            self.args.dev_file = self.args.qa_dev_file
        else:
            raise NotImplementedError
    
    def clear_biencoder_and_dev_file(self):
        #for code safety
        self.biencoder = None
        self.args.dev_file = None

    def validate_and_save_by_task(self, epoch: int, iteration: int, task_name: str, scheduler):

        self.set_biencoder_and_dev_file_by_task(task_name)
        self.validate_and_save(epoch, iteration, scheduler) 
        self.clear_biencoder_and_dev_file()


    def validate_average_rank_by_task(self, task_name: str):
        '''the only inputs to this function is dev_file and biencoder'''

        self.set_biencoder_and_dev_file_by_task(task_name)
        self.validate_average_rank()
        self.clear_biencoder_and_dev_file()

    def validate_reranking_by_task(self, task_name: str) -> float:

        self.set_biencoder_and_dev_file_by_task(task_name)
        self.validate_reranking()
        self.clear_biencoder_and_dev_file()

    def validate_nll_by_task(self, task_name: str) -> float:
        self.set_biencoder_and_dev_file_by_task(task_name)
        self.validate_nll()
        self.clear_biencoder_and_dev_file()        

        

def main():
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_training_params(parser)
    add_tokenizer_params(parser)

    # biencoder specific training features
    parser.add_argument("--eval_per_epoch", default=1, type=int,
                        help="How many times it evaluates on dev set per epoch and saves a checkpoint")

    parser.add_argument("--global_loss_buf_sz", type=int, default=150000,
                        help='Buffer size for distributed mode representations al gather operation. \
                                Increase this if you see errors like "encoded data exceeds max_size ..."')

    parser.add_argument("--fix_ctx_encoder", action='store_true')
    parser.add_argument("--shuffle_positive_ctx", action='store_true')

    #added options
    parser.add_argument("--share_encoder", action='store_true', help='whether two encoders are shared')
    parser.add_argument("--which_encoder_to_load", type=str, default='c', 
                        help='when the encoder is shared, which encoder to load from pretrained open domain QA model.\
                            there are only two options: q meaning question encoder, c meaning document encoder')
    parser.add_argument("--use_title_in_ctx", action='store_true', help='whether title in the ctx is used')
    parser.add_argument("--model_selection_task_name", type=str, default='qqs', help='which task to use for model selection.\
                                                                                    choose can be qa, aas')
    # parser.add_argument("--model_selection_metric", type=str, default='avg_rank', help='options are: avg_rank, loss, map')
    parser.add_argument("--train_or_test", type=str, default='test', help='options are: train, test')

    # input/output src params
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model checkpoints will be written or resumed from")

    # data handling parameters
    parser.add_argument("--hard_negatives", default=10, type=int,
                        help="amount of hard negative ctx per question")
    parser.add_argument("--other_negatives", default=0, type=int,
                        help="amount of 'other' negative ctx per question")
    parser.add_argument("--train_files_upsample_rates", type=str,
                        help="list of up-sample rates per each train file. Example: [1,2,1]")

    # parameters for Av.rank validation method
    parser.add_argument("--val_av_rank_start_epoch", type=int, default=1,
                        help="Av.rank validation: the epoch from which to enable this validation")
    parser.add_argument("--val_av_rank_hard_neg", type=int, default=30,
                        help="Av.rank validation: how many hard negatives to take from each question pool")
    parser.add_argument("--val_av_rank_other_neg", type=int, default=30,
                        help="Av.rank validation: how many 'other' negatives to take from each question pool")
    parser.add_argument("--val_av_rank_bsz", type=int, default=128,
                        help="Av.rank validation: batch size to process passages")
    parser.add_argument("--val_av_rank_max_qs", type=int, default=10000,
                        help="Av.rank validation: max num of questions")
    parser.add_argument('--checkpoint_file_name', type=str, default='dpr_biencoder', help="Checkpoints file prefix")


    '''
    multiple task specific parameters.
    Here, the train_file and dev_file will not be used
    '''
    parser.add_argument("--question_sim_train_file", default=None, type=str, help="File pattern for the train set. \
                        If the file is given, question similarity task will be trained")
    parser.add_argument("--question_sim_dev_file", default=None, type=str, help="If the file is given, \
                        question similarity task will be tested")

    parser.add_argument("--answer_sim_train_file", default=None, type=str, help="File pattern for the train set. \
                        If the file is given, answer similarity task will be trained")
    parser.add_argument("--answer_sim_dev_file", default=None, type=str, help="If the file is given, \
                        answer similarity task will be tested")

    parser.add_argument("--qa_train_file", default=None, type=str, help="File pattern for the train set. \
                        If the file is given, qa task will be trained")
    parser.add_argument("--qa_dev_file", default=None, type=str, help="If the file is given, qa task will be tested")
    # parser.add_argument("--num_iterations_per_epoch", default=-1, type=int)

    parser.add_argument('--num_qa_data_repetition', default=1, type=int, help='1')
    parser.add_argument('--num_qqs_data_repetition', default=3, type=int, help='semeval=20, askubuntu=3')
    parser.add_argument('--num_aas_data_repetition', default=0, type=int, help='not used')

    args = parser.parse_args()

    #test
    if args.question_sim_train_file:
        assert args.question_sim_dev_file
    if args.answer_sim_train_file:
        assert args.answer_sim_dev_file
    if args.qa_train_file:
        assert args.qa_dev_file


    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    setup_args_gpu(args)
    set_seed(args)
    print_args(args)
    # breakpoint()
    #options: qa, qqs. aas
    # eval_task_name = 'qqs'


    trainer = BiEncoderTrainerMultiTask(args)
    if args.train_or_test == 'train':
        assert args.question_sim_train_file is not None or args.answer_sim_train_file is not None or args.qa_train_file is not None, 'training files are missing'
        trainer.run_train(args.model_selection_task_name)
        logger.info("evaluate using the last iteration")
        trainer.validate_reranking_by_task(args.model_selection_task_name)
        trainer.validate_average_rank_by_task(args.model_selection_task_name)
    elif args.train_or_test == 'test': 
        assert args.model_file and (args.question_sim_dev_file or args.answer_sim_dev_file or args.qa_dev_file), 'test files are missing'
        # logger.info("No train files are specified. Run 2 types of validation for specified model file")
        trainer.validate_reranking_by_task(args.model_selection_task_name)
        # trainer.validate_nll_by_task(eval_task_name)
        trainer.validate_average_rank_by_task(args.model_selection_task_name)
    else:
        raise NotImplementedError("train_or_test not set correctly")


if __name__ == "__main__":
    main()