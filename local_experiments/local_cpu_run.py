import os
# os.environ["PYTORCH_ENABLE_MPS"] = "0"  # 必须在所有导入之前
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 禁用 CUDA
# os.environ["NO_MPS"] = "1"  # 额外保险
from swift.pipelines import sft_main
from swift.arguments import SftArguments
if __name__=="__main__":
    result = sft_main(SftArguments(
        model='/Users/guoxing.lan/projects/models/qwen3/Qwen3-0.6B',
        tuner_type='full',
        dataset=[
            # '/Users/guoxing.lan/projects/datasets/LLM/sft/wuzika_planet/train'
            '/Users/guoxing.lan/projects/datasets/LLM/sft/swift_encode_examples/agent_sft_user_multi_turns2.json'
                 ],
        val_dataset=[
            # '/Users/guoxing.lan/projects/datasets/LLM/sft/wuzika_planet/val'
            '/Users/guoxing.lan/projects/datasets/LLM/sft/swift_encode_examples/agent_sft_parallel_multi_turns4.json'

        ],
        loss_scale="last_round",
        torch_dtype='float32',
        num_train_epochs=1,
        packing=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-5,
        gradient_accumulation_steps=1,
        eval_steps=1,
        save_steps=1,
        save_total_limit=2,
        logging_steps=1,
        max_length=512,
        # use_mps_device=False,
        output_dir='outputs/0.6B_agent1',
        use_cpu=True,
        device_map='cpu',
        warmup_ratio=0.05,
        dataloader_num_workers=0,
        # no_cuda=True,
        gradient_checkpointing=True,
        optim='adamw_torch',
        max_grad_norm=1.0,

    ))
    print(result)