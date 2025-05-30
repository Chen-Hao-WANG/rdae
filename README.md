# src
source tf-gpu/bin/activate
# tensor board
tensorboard --logdir /home/howard/rdae/output_folder/summaries
# Training Command
python main.py \
    --dataPath /home/howard/rdae/data_folder \
    --outputPath /home/howard/rdae/output_folder \
    --model AE \
    --seqLen 1 \
    --inpChannel 3 \
    --originalH 768 \
    --originalW 1024 \
    --nepoch 50 \
    --batchSize 1 \
    --learningRate 1e-4 \
    --saveImgFreq 5 \
    --saveModelFreq 10 \
    --store
# Prediction Command
python main.py \
    --dataPath /home/howard/rdae/data_folder \
    --outputPath /home/howard/rdae/output_folder/predict_results_png_target \
    --model AE \
    --seqLen 1 \
    --inpChannel 3 \
    --originalH 768 \
    --originalW 1024 \
    --batchSize 1 \
    --phase predict \
    --isRefine \
    --pretrainPath /home/howard/rdae/output_folder/model_weights/AE_best.weights.h5

