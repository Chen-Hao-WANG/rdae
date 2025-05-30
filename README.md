# src
source tf-gpu/bin/activate
# tensor board
tensorboard --logdir /home/howard/rdae/output_folder/summaries
# Training Command
```
python main.py
    --dataPath ./data_folder
    --outputPath ./output_folder
    --model AE
    --seqLen 1
    --inpChannel 3
    --originalH 768
    --originalW 1024
    --nepoch 50
    --batchSize 1
    --learningRate 1e-4
    --saveImgFreq 5
    --saveModelFreq 10
    --store
```
```
python main.py \
    --dataPath ./data_folder \
    --outputPath ./output_folder \
    --model AE \
    --seqLen 1 \
    --inpChannel 3 \
    --originalH 768 \
    --originalW 1024 \
    --nepoch 50 \
    --batchSize 8  # <--- 建議從這個值開始嘗試，可以根據 VRAM 使用情況調整 (例如 4, 8, 16, 32)
    --learningRate 1e-4 \
    --saveImgFreq 50   # <--- 配合 batchSize 增加，可以適當增加儲存頻率，避免過於頻繁
    --saveModelFreq 100  # <--- 同上
    --store \
    --gpuId 0 \
    --manualSeed 42 \
    --earlyStoppingPatience 10 \
    --earlyStoppingMinDelta 0.0001 \
    --ws 0.8 \
    --wg 0.1 \
    --wt 0.1 # 如果是 AE 模型且 seqLen=1，wt 參數不會產生作用
```
# Prediction Command
```
python main.py \
    --dataPath ./data_folder \
    --outputPath ./predict_results_png_target \
    --model AE \
    --seqLen 1 \
    --inpChannel 3 \
    --originalH 768 \
    --originalW 1024 \
    --batchSize 1 \
    --phase predict \
    --isRefine \
    --pretrainPath ./output_folder/model_weights/AE_best.weights.h5
```
```
python main.py \
    --dataPath ./data_folder_for_prediction \
    --outputPath ./predict_results \
    --model AE \
    --seqLen 1 \
    --inpChannel 3 \
    --originalH 768 \
    --originalW 1024 \
    --batchSize 4  # <--- 預測時也可以適當增加 batchSize 以加快速度，但通常不需要像訓練時那麼大
    --phase predict \
    --isRefine \
    --pretrainPath ./output_folder/model_weights/AE_best.weights.h5 \
    --gpuId 0
```

