import warnings
from math import ceil
from accelerate.utils import set_seed
from torch.utils.data.dataloader import DataLoader
from lm_eval.utils import EndOfFunctionCriteria, TokenizedDataset, complete_code
from watermark import WatermarkLogitsProcessor
from transformers import StoppingCriteriaList, LogitsProcessorList
from lm_eval import tasks
import pdb
import torch,json
from tqdm import tqdm
from watermark import WatermarkDetector
import os
import numpy as np
class Generator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

    def generate_text(self, task_name):
        task = tasks.get_task(task_name)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        n_tasks = self.args.limit if self.args.limit else len(dataset)
        generations = self.parallel_generations(
            task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
        )
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]
        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references
    
    def parallel_generations(self, task, dataset, accelerator, model, tokenizer, n_tasks, args):
        if args.load_generations_path:
        # load generated code
            with open(args.load_generations_path) as fp:
                generations = json.load(fp)
                if accelerator.is_main_process:
                    print(
                        f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
                    )
            return generations[:n_tasks]
        
        set_seed(args.seed, device_specific=True)

        # Setup generation settings
        gen_kwargs = {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_length": args.max_length_generation,
        }
        if task.stop_words:
            if tokenizer.eos_token:
                task.stop_words.append(tokenizer.eos_token)
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [EndOfFunctionCriteria(0, task.stop_words, tokenizer)]
            )

        watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                            gamma=args.gamma,
                                                            delta=args.delta)
        gen_kwargs["logits_processor"] = LogitsProcessorList(
                [watermark_processor]
            )

        if accelerator.is_main_process:
            print(f"number of problems for this task is {n_tasks}")
        n_copies = ceil(args.n_samples / args.batch_size)

        ds_tokenized = TokenizedDataset(
            task,
            dataset,
            tokenizer,
            num_devices=accelerator.state.num_processes,
            max_length=args.max_length_generation,
            n_tasks=n_tasks,
            n_copies=n_copies,
            prefix=args.prefix,
        )

        # do not confuse args.batch_size, which is actually the num_return_sequences
        ds_loader = DataLoader(ds_tokenized, batch_size=1)
        model = model.to(accelerator.device)

        ds_loader = accelerator.prepare(ds_loader)

        generations = complete_code(
            task,
            accelerator,
            model,
            tokenizer,
            ds_loader,
            n_tasks=n_tasks,
            batch_size=args.batch_size,
            prefix=args.prefix,
            preprocess=False,
            postprocess=args.postprocess,
            **gen_kwargs,
        )
        return generations
    
    def evaluate(self, task_name):
        task = tasks.get_task(task_name)
        if self.args.detect_human_code:
            task_dataset = task.get_dataset()
            
            generations = []
            for i in range(len(task_dataset)):
                full_human = task.get_full_data(task_dataset[i])
                if full_human:
                    generations.append([full_human])

            references = [task.get_reference(task_dataset[i]) for i in range(len(task_dataset))]
        else:
            generations, references = self.generate_text(task_name)
        wllm_detector = WatermarkDetector(vocab=list(self.tokenizer.get_vocab().values()),
                                        gamma=self.args.gamma,
                                        tokenizer=self.tokenizer,type='wllm',
                                        model=None,acc=None)
        sweet_detector = WatermarkDetector(vocab=list(self.tokenizer.get_vocab().values()),
                                        gamma=self.args.gamma,
                                        tokenizer=self.tokenizer,type='sweet',
                                        model=self.model,acc=self.accelerator)
        if self.args.task =='mbpp':
            sweet_detector.entropy_threshold=0.6953125
        elif self.args.task =='humaneval':
            sweet_detector.entropy_threshold=0.6796875
        ewd_detector = WatermarkDetector(vocab=list(self.tokenizer.get_vocab().values()),
                                        gamma=self.args.gamma,
                                        tokenizer=self.tokenizer,type='ewd',
                                        model=self.model,acc=self.accelerator)
        print('Using kgw watermark detector...')
        wllm_detection_results = self.detect_watermark(task_name, generations, wllm_detector)
        print('Using sweet watermark detector...')
        sweet_detection_results = self.detect_watermark(task_name, generations, sweet_detector)
        print('Using ewd watermark detector...')
        ewd_detection_results = self.detect_watermark(task_name, generations, ewd_detector)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if self.allow_code_execution and task.requires_execution:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        try:
            results, pass_info = task.process_results(generations, references)
        except:
            results,pass_info={},'something went wrong during code evaluation'

        results["wllm_detection_results"] = wllm_detection_results
        results["sweet_detection_results"] = sweet_detection_results
        results["ewd_detection_results"] = ewd_detection_results

        results["pass_info"] = pass_info

        entropy_all=[]
        for i in sweet_detection_results:
            entropy_all+=i['entropy']
        q1 = np.percentile(entropy_all, 25)
        median = np.median(entropy_all)
        q3 = np.percentile(entropy_all, 75)
        print('q1:',q1,'q3:',q3,'median:',median)
        return results


    def detect_watermark(self, task_name, generations,watermark_detector):
        task = tasks.get_task(task_name)
        dataset = task.get_dataset()
        n_tasks = self.args.limit if self.args.limit else len(generations)
        def tokenize(example):
            inputs = self.tokenizer(
                example,
                padding=True,
                truncation=True,
                return_tensors="pt",
                #max_length=args.max_length_generation,
            )
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
        prompt_contents = [task.get_prompt(dataset[sample]) for sample in range(n_tasks)]
        detection_results=[]
        if self.accelerator.is_main_process:
            for idx, gens in tqdm(enumerate(generations), total=n_tasks):
                if idx >= n_tasks:
                    break
                n_detection = 0
                for idx2, gen in enumerate(gens):
                    # we don't check all n_samples generations
                    if n_detection >= 1:
                        continue
                    prefix = prompt_contents[idx]
                    tokenized_prefix = tokenize(prefix)['input_ids'].squeeze()
                    prefix_len = len(tokenized_prefix)
                    # if the prompt is not part of generation
                    try:
                        assert gen.startswith(prefix)
                    except AssertionError:
                        print(f"{idx}, {idx2}")
                        pdb.set_trace()
                    tokenized_text = tokenize(gen)['input_ids'].squeeze()
                    # if tokenized are not same
                    try:
                        assert torch.equal(tokenized_text[:prefix_len],tokenized_prefix), "Tokenized prefix must be a prefix of the tokenized text"
                    except AssertionError:
                        print(f"{idx}, {idx2}")
                        # tokenizing issue.. check at least the lens are same
                        if len(tokenized_text[:prefix_len]) == len(tokenized_prefix):
                            pass
                        else:
                            pdb.set_trace()
                    # if len of generation is 0, check next genertion
                    if len(tokenized_text) - prefix_len == 0:
                        continue
                    else:
                        if idx2 != 0:
                            print(idx2)
                        n_detection += 1
                    detection_result = watermark_detector.detect(
                            tokenized_text=tokenized_text,
                            tokenized_prefix=tokenized_prefix,
                        )
                    if not detection_result.pop('invalid', False):
                            detection_results.append(detection_result)
                if n_detection < 1:
                    print(f"all {idx}th generations are 0 len.")
        return detection_results        


