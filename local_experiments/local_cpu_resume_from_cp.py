import os
# os.environ["PYTORCH_ENABLE_MPS"] = "0"  # 必须在所有导入之前
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 禁用 CUDA
# os.environ["NO_MPS"] = "1"  # 额外保险
from swift.llm import sft_main, TrainArguments
if __name__=="__main__":
    checkpoint_path=r"D:\projects\github\ms-swift\outputs\0.6B_self_cognition_mixin1\v0-20250704-201055\checkpoint-4"
    result = sft_main(TrainArguments(
        model=r'D:\projects\models\qwen3\Qwen3-0.6B',
        train_type='full',
        dataset=[r'D:\projects\datasets\self-cognition\self_cognition_train.jsonl',
                 r'D:\projects\datasets\Qwen3-SFT-Mixin\qwen3_32b_distill_1k_train.jsonl'
                 ],
        val_dataset=[r'D:\projects\datasets\self-cognition\self_cognition_val.jsonl',
                 r'D:\projects\datasets\Qwen3-SFT-Mixin\qwen3_32b_distill_1k_val.jsonl'
                 ],
        torch_dtype='float32',
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-5,
        gradient_accumulation_steps=1,
        eval_steps=4,
        save_steps=4,
        save_total_limit=2,
        logging_steps=1,
        max_length=512,
        use_mps_device=False,
        output_dir='outputs/0.6B_self_cognition_mixin1',
        use_cpu=True,
        device_map='cpu',
        warmup_ratio=0.05,
        dataloader_num_workers=1,
        no_cuda=True,
        gradient_checkpointing=True,
        optim='adamw_torch',
        max_grad_norm=1.0,
        resume_from_checkpoint=checkpoint_path

    ))
    print(result)