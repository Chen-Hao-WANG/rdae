import tensorflow as tf
# 在所有 TensorFlow 操作之前設定全域混合精度策略
# 確保這行在腳本的非常靠前的位置
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("已嘗試設定全域混合精度策略為 'mixed_float16'")
except Exception as e:
    print(f"設定全域混合精度策略時發生錯誤: {e}")
    print("請確保 TensorFlow 版本支援混合精度。")

import numpy as np
import os
import shutil # 用於清理資料夾

# 假設你的原始檔案在同一個專案結構中
# 根據你的實際路徑調整這些導入
try:
    import utils    # 從 chen-hao-wang/rdae/rdae-de2a1a8184d1321cd7a09bb7610c1c205bf65ccf/utils.py
    import rae      # 從 chen-hao-wang/rdae/rdae-de2a1a8184d1321cd7a09bb7610c1c205bf65ccf/rae.py
except ImportError as e:
    print(f"導入模組時發生錯誤: {e}")
    print("請確保此腳本與你的 utils.py, rae.py 在正確的 Python 路徑下，或者調整導入語句。")
    exit()

def check_model_forward_pass_memory(batch_size, height, width, channels):
    print(f"\n開始檢查模型前向傳播記憶體佔用情況...")
    print(f"設定: Batch Size={batch_size}, Height={height}, Width={width}, Channels={channels}")
    current_policy = tf.keras.mixed_precision.global_policy()
    print(f"當前 TensorFlow 全域 Dtype Policy: Name={current_policy.name}, Compute Dtype={current_policy.compute_dtype}, Variable Dtype={current_policy.variable_dtype}")


    # 配置 GPU (與你的 main.py 類似)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"正在使用 GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(f"GPU 配置錯誤: {e}")
            return
    else:
        print("未找到 GPU，將在 CPU 上運行 (記憶體檢查意義不大)。")

    dummy_summary_dir = "./dummy_summaries_for_oom_check"
    os.makedirs(dummy_summary_dir, exist_ok=True)

    model_instance = None
    try:
        model_instance = rae.RAE(
            model_type_str='AE',
            inpChannel=channels,
            model_input_h=height,
            model_input_w=width,
            batchSize=batch_size,
            seqLen=1,
            learning_rate=1e-4,
            ws=0.8, wg=0.1, wt=0.0,
            phase='PREDICT',
            sum_dir=dummy_summary_dir
        )
        print("AE 模型初始化成功。")
        print(f"模型實例 '{model_instance.name}' 的 Dtype Policy: {model_instance.dtype_policy.name}")
        print(f"  Compute Dtype: {model_instance.compute_dtype}, Variable Dtype: {model_instance.variable_dtype}")


        dummy_input_tensor = tf.zeros((batch_size, height, width, channels), dtype=tf.float32)
        # 即使全域策略是 mixed_float16，輸入資料通常仍是 float32，模型內部會進行轉換
        # 或者，你可以明確地將輸入轉換為模型的 compute_dtype，但通常 Keras 會自動處理
        # dummy_input_tensor = tf.cast(dummy_input_tensor, model_instance.compute_dtype)
        model_input_list = [dummy_input_tensor]
        print(f"建立虛擬輸入張量，shape: {dummy_input_tensor.shape}, dtype: {dummy_input_tensor.dtype}")

        print("嘗試執行一次模型前向傳播 (model build)...")
        _ = model_instance(model_input_list, training=False)
        print("模型首次前向傳播 (build) 成功。")

        print("\n--- Encoder Model 層級 Dtype Policy 檢查 ---")
        for layer in model_instance.encoder_model.layers:
            print(f"  Encoder Layer: {layer.name:<25} Dtype Policy: {layer.dtype_policy.name:<15} Compute Dtype: {layer.compute_dtype:<10} Variable Dtype: {layer.variable_dtype}")
        
        print("\n--- Decoder Model 層級 Dtype Policy 檢查 ---")
        for layer in model_instance.decoder_model.layers:
            print(f"  Decoder Layer: {layer.name:<25} Dtype Policy: {layer.dtype_policy.name:<15} Compute Dtype: {layer.compute_dtype:<10} Variable Dtype: {layer.variable_dtype}")

        print("\n--- Encoder Model Summary ---")
        model_instance.encoder_model.summary(print_fn=lambda x: print(x, flush=True)) #確保即時打印
        print("\n--- Decoder Model Summary ---")
        model_instance.decoder_model.summary(print_fn=lambda x: print(x, flush=True))

        print("\n再次嘗試執行模型前向傳播...")
        output_tensor_list = model_instance(model_input_list, training=False)
        output_tensor = output_tensor_list[0]
        print(f"模型前向傳播成功，輸出 shape: {output_tensor.shape}, 輸出 dtype: {output_tensor.dtype}")
        # 當使用混合精度時，模型的輸出 dtype 通常會是 float16 (如果最後一層參與了混合精度計算)
        # 或者由最後一層的 dtype policy 決定 (例如，如果最後一層被明確設定為 float32 輸出以保證精度)
        # 你的模型最後一層 Dec_conv_6 的 activation 是 None，它會遵循模型的 compute_dtype (float16)
        
        print("注意：這僅測試了前向傳播。訓練時反向傳播需要更多記憶體。")

    except tf.errors.ResourceExhaustedError as e:
        print("\n捕獲到 ResourceExhaustedError (OOM)!")
        print(f"錯誤訊息: {e}")
        print("這表明即使只是前向傳播（或模型建構時的記憶體分配），對於當前配置也記憶體不足。")
        print("建議嘗試以下一種或多種方法：")
        print("  1. 在實際訓練中減小批次大小 (batchSize)。")
        print("  2. 減小模型輸入的圖像尺寸 (height, width)。")
        print("  3. 簡化模型結構 (減少 utils.py 中模型的濾鏡數量或層數)。")
        print("  4. (如果尚未在訓練中啟用)在實際訓練中啟用混合精度訓練。")
        print("  5. 在實際訓練前設定環境變量 TF_GPU_ALLOCATOR=cuda_malloc_async。")
    except Exception as e:
        print(f"執行過程中發生其他錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(dummy_summary_dir):
            try:
                shutil.rmtree(dummy_summary_dir)
                print(f"已清理臨時資料夾: {dummy_summary_dir}")
            except Exception as e:
                print(f"清理臨時資料夾失敗: {e}")

if __name__ == "__main__":
    check_batch_size = 4
    check_height = 768
    check_width = 1024
    check_channels = 3

    check_model_forward_pass_memory(check_batch_size, check_height, check_width, check_channels)
