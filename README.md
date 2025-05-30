# src
source tf-gpu/bin/activate
# tensor board
tensorboard --logdir ./rdae/output_folder/summaries
# Training Command
```
python main.py --dataPath ./data_folder --outputPath ./output_folder --model AE --seqLen 1 --inpChannel 3 --originalH 768 --originalW 1024 --nepoch 5 --batchSize 1 --learningRate 1e-4 --saveImgFreq 10 --saveModelFreq 20 --store

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
    --batchSize 8  
    --learningRate 1e-4 \
    --saveImgFreq 50   
    --saveModelFreq 100  
    --store \
    --gpuId 0 \
    --manualSeed 42 \
    --earlyStoppingPatience 10 \
    --earlyStoppingMinDelta 0.0001 \
    --ws 0.8 \
    --wg 0.1 \
    --wt 0.1 
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

# check pipline
```
python check_pipeline.py \
    --dataPath ./data_folder \
    --outputDir ./output_folder \
    --originalH 768 \
    --originalW 1024 \
    --inpChannel 3 \
    --batchSize 1 \
    --pretrainPath ./output_folder/model_weights/AE_best.weights.h5

```

# check OOM
```
python check_oom.py
```

# increase mem設定 TensorFlow GPU 分配器環境變數
```
export TF_GPU_ALLOCATOR=cuda_malloc_async
```