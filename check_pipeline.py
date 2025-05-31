import os
import argparse
import numpy as np
import imageio
import tensorflow as tf
from PIL import Image

# 假設你的原始檔案在同一個專案結構中
# 根據你的實際路徑調整這些導入
try:
    import dataLoader # 從 chen-hao-wang/rdae/rdae-de2a1a8184d1321cd7a09bb7610c1c205bf65ccf/dataLoader.py
    import rae      # 從 chen-hao-wang/rdae/rdae-de2a1a8184d1321cd7a09bb7610c1c205bf65ccf/rae.py
    import utils    # 從 chen-hao-wang/rdae/rdae-de2a1a8184d1321cd7a09bb7610c1c205bf65ccf/utils.py
except ImportError as e:
    print(f"導入模組時發生錯誤: {e}")
    print("請確保 check_pipeline.py 與你的 dataLoader.py, rae.py, utils.py 在正確的 Python 路徑下，或者調整導入語句。")
    exit()

def denormalize_image(normalized_img_tensor):
    """
    將歸一化到 [-1, 1] 的圖像張量反歸一化回 [0, 255] 的 uint8 numpy 陣列。
    """
    if isinstance(normalized_img_tensor, tf.Tensor):
        img_np = normalized_img_tensor.numpy()
    else:
        img_np = normalized_img_tensor

    # 反歸一化：從 [-1, 1] 到 [0, 2] 再到 [0, 255]
    # (X * 128) + 128
    img = np.add(np.multiply(img_np, 128.0), 128.0)
    return np.clip(img, 0, 255).astype(np.uint8)

def main(opt):
    print("開始執行管線檢查腳本...")
    os.makedirs(opt.outputDir, exist_ok=True)
    print(f"檢查結果將儲存於: {opt.outputDir}")

    # --- 1. 檢查資料載入與預處理 ---
    print("\n--- 步驟 1: 檢查資料載入與預處理 ---")
    try:
        loader = dataLoader.BatchLoader(
            dataRoot=opt.dataPath,
            inpChannel=opt.inpChannel,
            batchSize=opt.batchSize, # 通常設為 1 以便檢查單個樣本
            seqLen=1, # AE 模型 seqLen 為 1
            scaleSize=(opt.originalH, opt.originalW),
            cropSize=(opt.originalH, opt.originalW),
            isRandom=False, # 檢查時通常不需要隨機
            phase='TEST', # 或 'PREDICT'，避免訓練時的隨機打亂
            rseed=42,
            process_original_size=True
        )
        print("BatchLoader 初始化成功。")
    except Exception as e:
        print(f"初始化 BatchLoader 失敗: {e}")
        return

    if loader.count == 0:
        print(f"錯誤: 在 {opt.dataPath} 中未找到任何圖像資料。請檢查路徑和資料。")
        return

    # 載入一個批次的資料
    # inputNoise_norm: 歸一化後的噪聲輸入 (模型實際接收的)
    # vizNoise_rgb: 未歸一化的噪聲輸入 (用於視覺化比較)
    # inputClean_norm: 歸一化後的乾淨目標
    # vizClean_rgb: 未歸一化的乾淨目標 (用於視覺化比較)
    inputNoise_norm_batch, vizNoise_rgb_batch, inputClean_norm_batch, vizClean_rgb_batch, _ = loader.loadBatch()

    if inputNoise_norm_batch.shape[0] == 0:
        print("錯誤: BatchLoader 未能載入任何資料。")
        return
    
    # 取第一個樣本進行檢查 (如果 batchSize > 1)
    sample_idx = 0
    inputNoise_norm_sample = inputNoise_norm_batch[sample_idx]
    vizNoise_rgb_sample = vizNoise_rgb_batch[sample_idx]
    inputClean_norm_sample = inputClean_norm_batch[sample_idx]
    vizClean_rgb_sample = vizClean_rgb_batch[sample_idx]

    print(f"已載入一個樣本。噪聲輸入 (歸一化後) shape: {inputNoise_norm_sample.shape}, dtype: {inputNoise_norm_sample.dtype}")
    print(f"噪聲輸入 (歸一化後) 數值範圍: Min={np.min(inputNoise_norm_sample):.4f}, Max={np.max(inputNoise_norm_sample):.4f}, Mean={np.mean(inputNoise_norm_sample):.4f}")
    print(f"乾淨目標 (歸一化後) shape: {inputClean_norm_sample.shape}, dtype: {inputClean_norm_sample.dtype}")
    print(f"乾淨目標 (歸一化後) 數值範圍: Min={np.min(inputClean_norm_sample):.4f}, Max={np.max(inputClean_norm_sample):.4f}, Mean={np.mean(inputClean_norm_sample):.4f}")

    # 儲存視覺化結果
    try:
        imageio.imwrite(os.path.join(opt.outputDir, "check_01_viz_noisy_input.png"), vizNoise_rgb_sample)
        print("已儲存: check_01_viz_noisy_input.png (原始噪聲圖，來自 dataLoader)")

        denorm_noisy_input = denormalize_image(inputNoise_norm_sample)
        imageio.imwrite(os.path.join(opt.outputDir, "check_02_denormalized_noisy_input.png"), denorm_noisy_input)
        print("已儲存: check_02_denormalized_noisy_input.png (歸一化後再反歸一化的噪聲圖)")

        imageio.imwrite(os.path.join(opt.outputDir, "check_03_viz_clean_target.png"), vizClean_rgb_sample)
        print("已儲存: check_03_viz_clean_target.png (原始乾淨目標圖，來自 dataLoader)")

        denorm_clean_target = denormalize_image(inputClean_norm_sample)
        imageio.imwrite(os.path.join(opt.outputDir, "check_04_denormalized_clean_target.png"), denorm_clean_target)
        print("已儲存: check_04_denormalized_clean_target.png (歸一化後再反歸一化的乾淨目標圖)")
    except Exception as e:
        print(f"儲存資料載入檢查圖片時出錯: {e}")


    # --- 2. 檢查模型初始化與權重載入 ---
    print("\n--- 步驟 2: 檢查模型初始化與權重載入 ---")
    # 確保使用 AE 模型
    model_type = 'AE'
    current_phase_upper = 'PREDICT' # 或 'TEST'

    # 建立一個假的 summary directory，因為 RAE 初始化需要
    dummy_summary_dir = os.path.join(opt.outputDir, "dummy_summaries")
    os.makedirs(dummy_summary_dir, exist_ok=True)

    try:
        # 注意：RAE 初始化需要一些 main.py 中的 opt 參數，這裡使用預設或從 opt 傳入
        # 這裡的 learning_rate, ws, wg, wt 僅用於初始化，在預測時不直接影響單次前向傳播
        model = rae.RAE(
            model_type_str=model_type,
            inpChannel=opt.inpChannel,
            model_input_h=opt.originalH,
            model_input_w=opt.originalW,
            batchSize=opt.batchSize, # 模型內部可能需要 batchSize 資訊
            seqLen=1, # AE 模型
            learning_rate=1e-4, # 預測時不重要
            ws=0.8, wg=0.1, wt=0.0, # 預測時不重要
            phase=current_phase_upper,
            sum_dir=dummy_summary_dir
        )
        print(f"{model_type} 模型初始化成功。")

        # 嘗試建構模型 (build)
        # 模型期望的輸入是 [batch_size, H, W, C] 的張量列表 (即使 AE 只有一個元素)
        # 或者直接是 [batch_size, H, W, C] 的張量 (取決於 call 方法的實現)
        # 根據 rae.py 的 call 方法，AE 模式下，如果輸入是列表，它會取第一個元素
        # 如果是張量，則直接使用。為保險起見，傳入列表。
        dummy_build_input = tf.zeros((opt.batchSize, opt.originalH, opt.originalW, opt.inpChannel), dtype=tf.float32)
        _ = model([dummy_build_input], training=False) # 呼叫 call 方法來建構
        print("模型建構 (build) 成功。")

    except Exception as e:
        print(f"初始化或建構 {model_type} 模型失敗: {e}")
        return

    if opt.pretrainPath:
        if os.path.exists(opt.pretrainPath):
            try:
                model.load_weights(opt.pretrainPath)
                print(f"成功從 {opt.pretrainPath} 載入模型權重。")
            except Exception as e:
                print(f"從 {opt.pretrainPath} 載入權重失敗: {e}")
                print("將使用隨機初始化的權重進行後續檢查。")
        else:
            print(f"警告: 找不到指定的預訓練權重檔案: {opt.pretrainPath}")
            print("將使用隨機初始化的權重進行後續檢查。")
    else:
        print("未提供預訓練權重路徑，將使用隨機初始化的權重。")


    # --- 3. 檢查模型預測與輸出 ---
    print("\n--- 步驟 3: 檢查模型預測與輸出 ---")
    # 模型輸入需要是批次形式，並且是歸一化的
    # inputNoise_norm_sample 的 shape 是 (H, W, C)
    # 需要擴展成 (1, H, W, C) 作為批次大小為 1 的輸入
    model_input_tensor = tf.expand_dims(tf.convert_to_tensor(inputNoise_norm_sample, dtype=tf.float32), axis=0)
    print(f"準備好的模型輸入張量 shape: {model_input_tensor.shape}")

    try:
        # 呼叫模型進行預測
        # model.call() 預期輸入是一個列表，即使 AE 只有一個元素
        denoised_output_list = model([model_input_tensor], training=False)
        
        if not denoised_output_list:
            print("錯誤: 模型預測返回了空列表。")
            return
            
        denoised_tensor_batch = denoised_output_list[0] # AE 模式下，列表只有一個元素
        denoised_tensor_sample = denoised_tensor_batch[0] # 取出批次中的第一個 (也是唯一一個) 樣本
        print(f"模型原始輸出 (歸一化後) shape: {denoised_tensor_sample.shape}, dtype: {denoised_tensor_sample.dtype}")
        print(f"模型原始輸出 (歸一化後) 數值範圍: Min={tf.reduce_min(denoised_tensor_sample).numpy():.4f}, Max={tf.reduce_max(denoised_tensor_sample).numpy():.4f}, Mean={tf.reduce_mean(denoised_tensor_sample).numpy():.4f}")

        # 反歸一化並儲存
        # rae.py 中的 _normalize_for_saving_and_tensorboard 包含了反歸一化邏輯
        # 這裡我們直接使用之前定義的 denormalize_image
        denoised_image_uint8 = denormalize_image(denoised_tensor_sample)

        # 確保通道數正確 (例如，如果是單通道灰階，但儲存時需要 RGB)
        if denoised_image_uint8.shape[-1] == 1 and opt.inpChannel == 1: # 假設目標也是灰階
             pass # 保持單通道
        elif denoised_image_uint8.shape[-1] == 1 and opt.inpChannel == 3: # 模型輸出單通道，但原輸入是彩色
            print("警告: 模型輸出為單通道，但原始輸入為3通道。將複製通道以形成RGB圖像。")
            denoised_image_uint8 = np.concatenate([denoised_image_uint8]*3, axis=-1)
        elif denoised_image_uint8.shape[-1] != 3 and opt.inpChannel == 3 : # 其他不匹配情況
             print(f"警告: 模型輸出通道數 ({denoised_image_uint8.shape[-1]}) 與預期 RGB (3) 不符。嘗試修正。")
             if denoised_image_uint8.shape[-1] > 3:
                 denoised_image_uint8 = denoised_image_uint8[..., :3]
             # 其他情況可能需要更複雜的處理，這裡僅作簡單示例

        imageio.imwrite(os.path.join(opt.outputDir, "check_05_model_denoised_output.png"), denoised_image_uint8)
        print("已儲存: check_05_model_denoised_output.png (模型去噪輸出圖)")

    except Exception as e:
        print(f"模型預測或儲存輸出時出錯: {e}")
        import traceback
        traceback.print_exc()


    # --- (可選) 4. 檢查損失計算 ---
    print("\n--- 步驟 4: (可選) 檢查損失計算 ---")
    # 這部分需要模型和乾淨目標都在歸一化空間
    # inputClean_norm_sample 是 (H, W, C)
    # denoised_tensor_sample 是 (H, W, C)
    try:
        # 確保維度匹配，並轉換為 TensorFlow 張量
        target_for_loss = tf.convert_to_tensor(inputClean_norm_sample, dtype=tf.float32)
        output_for_loss = tf.convert_to_tensor(denoised_tensor_sample, dtype=tf.float32)

        # L1 損失 (空間損失)
        # 模型內部 _get_L1_loss 期望的是列表，這裡我們直接計算
        # 注意：模型內部的 _preprocess 函數目前是恆等映射
        l1_loss_val = tf.reduce_mean(tf.abs(output_for_loss - target_for_loss))
        print(f"計算得到的 L1 (空間) 損失: {l1_loss_val.numpy():.6f}")

        # 梯度損失
        # 需要使用模型內部的 _applyGrad 方法
        # _applyGrad 期望輸入是批次形式 [batch, H, W, C]
        grad_output = model._applyGrad(tf.expand_dims(output_for_loss, axis=0))[0] # 取出批次維度
        grad_target = model._applyGrad(tf.expand_dims(target_for_loss, axis=0))[0] # 取出批次維度
        grad_loss_val = tf.reduce_mean(tf.abs(grad_output - grad_target))
        print(f"計算得到的梯度損失: {grad_loss_val.numpy():.6f}")
        
        # 總損失 (根據 main.py 中的預設權重)
        ws_check, wg_check = 0.8, 0.1 # 與 opt.ws, opt.wg 對應
        total_loss_check = ws_check * l1_loss_val + wg_check * grad_loss_val
        print(f"根據預設權重計算的總損失 (ws={ws_check}, wg={wg_check}): {total_loss_check.numpy():.6f}")

    except Exception as e:
        print(f"計算損失時出錯: {e}")

    print("\n檢查腳本執行完畢。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="檢查圖像去噪管線的腳本")
    parser.add_argument('--dataPath', required=True, help='包含噪聲 PPM 圖像的資料夾路徑')
    parser.add_argument('--outputDir', required=True, help='儲存檢查結果圖像的資料夾路徑')
    parser.add_argument('--pretrainPath', default=None, help='(可選) 已訓練模型的權重檔案路徑 (.h5)')
    
    parser.add_argument('--originalH', type=int, required=True, help='圖像原始高度 (模型輸入高度)')
    parser.add_argument('--originalW', type=int, required=True, help='圖像原始寬度 (模型輸入寬度)')
    parser.add_argument('--inpChannel', type=int, default=3, help='輸入圖像通道數 (例如 3 代表 RGB)')
    parser.add_argument('--batchSize', type=int, default=1, help='用於載入資料的批次大小 (建議設為 1 進行檢查)')
    # 你可以根據需要添加更多來自 main.py 的 opt 參數，如果 RAE 或 dataLoader 初始化需要它們

    args = parser.parse_args()
    main(args)