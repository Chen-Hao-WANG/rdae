from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import imageio 
import tensorflow as tf
import time 
from tqdm import tqdm 
import random

from rae import RAE 
import dataLoader 

parser = argparse.ArgumentParser(description="TensorFlow 2.x RAE Model Training for Denoising")
parser.add_argument('--model', type=str, default='AE', choices=['AE', 'RAE'], help='Model type: AE or RAE')
parser.add_argument('--nepoch', type=int, default=50, help='Number of epochs for training')
parser.add_argument('--batchSize', type=int, default=1, help='Input batch size')
parser.add_argument('--learningRate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--inpChannel', type=int, default=3, help='Input channel number')
parser.add_argument('--seqLen', type=int, default=1, help='Length of training sequence (fixed to 1 for AE if AE model is chosen)')

parser.add_argument('--originalH', type=int, required=True, help='Original height of the PPM images')
parser.add_argument('--originalW', type=int, required=True, help='Original width of the PPM images')

parser.add_argument('--scaleW', type=int, default=0, help='(Not used if processing original size)') 
parser.add_argument('--scaleH', type=int, default=0, help='(Not used if processing original size)')
parser.add_argument('--cropW', type=int, default=0, help='(Not used if processing original size)')
parser.add_argument('--cropH', type=int, default=0, help='(Not used if processing original size)')

parser.add_argument('--store', action='store_true', help='Store weights')
parser.add_argument('--saveModelFreq', type=int, default=50, help='Iteration frequency to save model weights')
parser.add_argument('--saveImgFreq', type=int, default=5, help='Iteration frequency to save comparison images') 

parser.add_argument('--dataPath', required=True, help='Path to dataset')
parser.add_argument('--outputPath', required=True, help='Path to store results, summaries, and models')
parser.add_argument('--pretrainPath', default=None, help='Path to pretrained model weights')

parser.add_argument('--gpuId', type=int, default=0, help='ID of GPU to use for training')
parser.add_argument('--manualSeed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--isRefine', action='store_true', help='Whether to refine from a pretrained model')

parser.add_argument('--ws', type=float, default=0.8, help='Coefficient of spatial L1 loss')
parser.add_argument('--wg', type=float, default=0.1, help='Coefficient of gradient-domain L1 loss')
parser.add_argument('--wt', type=float, default=0.1, help='Coefficient of temporal L1 loss (only for RAE)')
parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'predict'], help='Phase')

parser.add_argument('--earlyStoppingPatience', type=int, default=10, help='Epochs to wait for improvement before stopping')
parser.add_argument('--earlyStoppingMinDelta', type=float, default=0.0001, help='Minimum change for improvement')

opt = parser.parse_args()

if opt.model == 'AE': 
    opt.seqLen = 1 # Enforce seqLen=1 for AE model
if opt.originalH <= 0 or opt.originalW <= 0:
    raise ValueError("--originalH and --originalW must be positive integers.")

if opt.manualSeed is not None:
    tf.random.set_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    random.seed(opt.manualSeed)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        target_gpu = gpus[opt.gpuId] if opt.gpuId < len(gpus) else (gpus[0] if len(gpus) > 0 else None)
        if target_gpu:
            tf.config.experimental.set_visible_devices(target_gpu, 'GPU')
            tf.config.experimental.set_memory_growth(target_gpu, True)
            print(f"正在使用 GPU: {target_gpu.name}")
        else:
            print("警告: 指定的 GPU ID 超出範圍且無可用 GPU。將在 CPU 上運行。")
    except RuntimeError as e: print(f"GPU 配置錯誤: {e}")
else: print("未找到 GPU，將在 CPU 上運行。")

image_results_directory = os.path.join(opt.outputPath, 'comparison_images')
summary_directory = os.path.join(opt.outputPath, 'summaries')
model_weights_directory = os.path.join(opt.outputPath, 'model_weights')
for p in [opt.outputPath, image_results_directory, summary_directory, model_weights_directory]:
    os.makedirs(p, exist_ok=True)

if __name__ == "__main__":
    current_phase_lower = opt.phase.lower()
    model_input_h, model_input_w = opt.originalH, opt.originalW
    
    loader = dataLoader.BatchLoader(
        dataRoot=opt.dataPath, inpChannel=opt.inpChannel,
        batchSize=opt.batchSize, seqLen=opt.seqLen, # seqLen is now consistently used
        scaleSize=(model_input_h, model_input_w), 
        cropSize=(model_input_h, model_input_w), 
        rseed=opt.manualSeed, phase=current_phase_lower,
        process_original_size=True 
    )
    print('數據加載器已初始化 (處理原始圖像尺寸)。')

    current_time_str = time.strftime("%Y%m%d-%H%M%S")
    log_dir_for_model = os.path.join(summary_directory, f"{opt.model}_{current_time_str}")

    rae_model = RAE(
        model_type_str=opt.model, 
        inpChannel=opt.inpChannel,
        model_input_h=model_input_h, model_input_w=model_input_w,
        batchSize=opt.batchSize, seqLen=opt.seqLen, # Pass seqLen to RAE
        learning_rate=opt.learningRate,
        ws=opt.ws, wg=opt.wg, wt=opt.wt,
        phase=opt.phase.upper(), sum_dir=log_dir_for_model
    )
    print(f"RAE Keras 模型 ({opt.model}) 已初始化，輸入尺寸 {model_input_h}x{model_input_w}。")

    dummy_input_shape_per_step = (opt.batchSize, model_input_h, model_input_w, opt.inpChannel)
    # RAE.call expects a list of tensors (sequence). For AE (seqLen=1), it's a list with one tensor.
    dummy_input_for_call = [tf.zeros(dummy_input_shape_per_step, dtype=tf.float32) for _ in range(opt.seqLen)]
    try:
        _ = rae_model(dummy_input_for_call) # Build call
        print("RAE 模型已通過虛擬批次成功構建。")
    except Exception as e:
        print(f"使用虛擬批次構建 RAE 模型時出錯: {e}")
        exit()

    model_weights_path = os.path.join(model_weights_directory, f"{opt.model}_latest.weights.h5")
    best_model_path = os.path.join(model_weights_directory, f"{opt.model}_best.weights.h5")
    load_path_attempt = None
    if opt.pretrainPath and os.path.exists(opt.pretrainPath): load_path_attempt = opt.pretrainPath
    elif opt.isRefine:
        if os.path.exists(best_model_path): load_path_attempt = best_model_path
        elif os.path.exists(model_weights_path): load_path_attempt = model_weights_path
    if load_path_attempt:
        try:
            rae_model.load_weights(load_path_attempt)
            print(f"成功從以下路徑加載權重: {load_path_attempt}")
        except Exception as e:
            print(f"從 {load_path_attempt} 加載權重時出錯: {e}。")
            if current_phase_lower == 'train': print("將從頭開始訓練。")
    elif opt.isRefine:
        print(f"警告: 指定了 --isRefine，但在 --pretrainPath、{best_model_path} 或 {model_weights_path} 未找到權重。")
        if current_phase_lower == 'train': print("將從頭開始訓練。")

    global_step_counter = 0
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop_triggered = False

    if current_phase_lower == 'train':
        print(f"開始訓練，最多 {opt.nepoch} 個 epoch...")
        for epoch in range(1, opt.nepoch + 1):
            if early_stop_triggered: break
            print(f"\nEpoch {epoch}/{opt.nepoch}")
            iters_per_epoch = loader.count // opt.batchSize + (1 if loader.count % opt.batchSize != 0 else 0)
            if loader.count == 0 : iters_per_epoch = 0
            epoch_pbar = tqdm(range(iters_per_epoch), desc=f"Epoch {epoch}")
            epoch_total_loss, num_batches_in_epoch = 0.0, 0

            for iter_in_epoch in epoch_pbar:
                # loadBatch now returns direct tensors:
                inputNoise_tensor, vizNoise_rgb_tensor, inputClean_tensor, vizClean_rgb_tensor, isNewEpoch_loader = loader.loadBatch()
                
                if inputNoise_tensor.shape[0] == 0 : 
                    if isNewEpoch_loader and iters_per_epoch > 0 : break 
                    continue
                
                # RAE.train_step expects data as (input_list, target_list)
                # For AE (seqLen=1), these lists will each contain one tensor.
                loss_metrics = rae_model.train_step(([inputNoise_tensor], [inputClean_tensor]))
                global_step_counter += 1
                current_iter_loss = loss_metrics['Total_loss'].numpy()
                epoch_total_loss += current_iter_loss
                num_batches_in_epoch += 1
                log_msg = f"Iter: {global_step_counter}"
                for name, val in loss_metrics.items(): log_msg += f", {name}: {val.numpy():.4f}"
                epoch_pbar.set_postfix_str(log_msg)

                if opt.store and global_step_counter % opt.saveModelFreq == 0:
                    iter_save_path = os.path.join(model_weights_directory, f"{opt.model}_iter_{global_step_counter}.weights.h5")
                    rae_model.save_weights(iter_save_path)
                    rae_model.save_weights(model_weights_path) 
                    print(f"\n模型權重已保存: {iter_save_path} & {model_weights_path}")
                
                if global_step_counter % opt.saveImgFreq == 0:
                    print(f'\n正在保存比較圖像 (全局步驟 {global_step_counter})...')
                    # save_comparison_images expects direct batch tensors
                    rae_model.save_comparison_images(
                        inputNoise_tensor, vizNoise_rgb_tensor, vizClean_rgb_tensor,
                        image_results_directory, epoch, global_step_counter, current_phase_lower
                    )
            
            if num_batches_in_epoch > 0:
                avg_epoch_loss = epoch_total_loss / num_batches_in_epoch
                print(f"Epoch {epoch} 平均總損失: {avg_epoch_loss:.4f}")
                if avg_epoch_loss < best_loss - opt.earlyStoppingMinDelta:
                    best_loss = avg_epoch_loss
                    epochs_no_improve = 0
                    if opt.store:
                        rae_model.save_weights(best_model_path)
                        print(f"新的最佳模型已保存到 {best_model_path}，損失為 {best_loss:.4f}")
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= opt.earlyStoppingPatience:
                    print(f"在 {epoch} 個 epoch 後觸發 Early stopping。")
                    early_stop_triggered = True
            if isNewEpoch_loader and epoch < opt.nepoch and not early_stop_triggered : 
                 print(f"數據加載器在 Keras epoch {epoch} 內發出新遍歷信號。")
    
    elif current_phase_lower in ['test', 'predict']:
        print(f"開始 {current_phase_lower} 階段...")
        if not load_path_attempt and not opt.pretrainPath :
             print(f"警告: 在沒有加載權重的情況下運行 {current_phase_lower} 階段。將使用隨機初始化的模型。")
        iters_for_phase = loader.count // opt.batchSize + (1 if loader.count % opt.batchSize != 0 else 0)
        if loader.count == 0 : iters_for_phase = 0
        phase_pbar = tqdm(range(iters_for_phase), desc=f"{current_phase_lower.capitalize()}")
        for iter_idx in phase_pbar:
            inputNoise_tensor, vizNoise_rgb_tensor, inputClean_tensor, vizClean_rgb_tensor, _ = loader.loadBatch()
            if inputNoise_tensor.shape[0] == 0: continue

            print(f'\n正在為 {current_phase_lower} 迭代 {iter_idx} 保存比較圖像...')
            rae_model.save_comparison_images(
                inputNoise_tensor, vizNoise_rgb_tensor, vizClean_rgb_tensor,
                image_results_directory, 0, iter_idx, current_phase_lower 
            )
        print(f"{current_phase_lower.capitalize()} 已完成。")
    print("處理完成。")
