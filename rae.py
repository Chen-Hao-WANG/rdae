from __future__ import absolute_import, division, print_function

import math
import os 
import time 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import imageio 
from tqdm import tqdm 
from PIL import Image 

from utils import create_encoder_cnn_model, create_encoder_rnn_model, create_decoder_model

class RAE(tf.keras.Model):

    def __init__(self, model_type_str, inpChannel, 
                 model_input_h, model_input_w, 
                 batchSize, seqLen, learning_rate, 
                 ws, wg, wt, phase, sum_dir, **kwargs):
        super(RAE, self).__init__(name='RAE_Model', **kwargs) 

        self.model_type_str = model_type_str
        self.inpChannel = inpChannel
        self.inpH = model_input_h 
        self.inpW = model_input_w 
        self.batch_size_static = batchSize 
        self.seqLen = seqLen # For AE, this is effectively 1
        self.ws = ws
        self.wg = wg
        self.wt = wt
        self.phase = phase.upper() 
        self.sum_dir = sum_dir 
        
        self.input_shape_single_step_cnn = (self.inpH, self.inpW, self.inpChannel)
        self.input_shape_sequence_rnn = (self.seqLen, self.inpH, self.inpW, self.inpChannel)

        leaky_relu_activation = tf.keras.layers.LeakyReLU(negative_slope=0.2)

        if self.model_type_str == 'RAE':
            self.encoder_model = create_encoder_rnn_model(
                self.input_shape_sequence_rnn, name="SequenceEncoderRNN"
            )
        elif self.model_type_str == "AE":
            self.encoder_model = create_encoder_cnn_model(self.input_shape_single_step_cnn, name="ImageEncoderCNN")
        else:
            raise ValueError(f"不支援的模型類型: {self.model_type_str}")

        s_h, s_w = self.inpH, self.inpW
        self.skip_connection_shapes_per_step = [
            (s_h, s_w, 32), (s_h // 2, s_w // 2, 43), (s_h // 4, s_w // 4, 57),
            (s_h // 8, s_w // 8, 76), (s_h // 16, s_w // 16, 101), (s_h // 32, s_w // 32, 101)
        ]
        for i_skip in range(len(self.skip_connection_shapes_per_step)):
            level_h, level_w, _ = self.skip_connection_shapes_per_step[i_skip]
            if level_h <= 0 or level_w <= 0:
                raise ValueError(
                    f"計算得到的跳接層維度在第 {i_skip+1} 層無效 "
                    f"(H: {level_h}, W: {level_w})，輸入圖像尺寸 HxW: {self.inpH}x{self.inpW}。"
                )
        print(f"為解碼器計算得到的跳接層形狀 (H, W, C): {self.skip_connection_shapes_per_step}")

        self.decoder_model = create_decoder_model(self.skip_connection_shapes_per_step, final_output_channels=3, name="SharedDecoder")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if self.phase == 'TRAIN':
            self.train_writer = tf.summary.create_file_writer(self.sum_dir)
            print(f"TensorBoard 日誌將保存在: {self.sum_dir}")
        else:
            self.train_writer = None
        self.leaky_relu = leaky_relu_activation

    def call(self, inputs, training=False):
        """
        inputs: 
            - If model_type_str == "AE": A single tensor [batch, H, W, C]
            - If model_type_str == "RAE": A list of tensors (or a stacked tensor [batch, seq, H, W, C]),
                                         representing the sequence.
                                         For consistency with how main.py prepares data for train_step,
                                         this method will expect a list for RAE as well.
                                         If a stacked tensor is passed for RAE, it will be unstacked.
        """
        if self.model_type_str == 'RAE':
            # Expects a list of tensors, one for each time step
            if not isinstance(inputs, list): # If a stacked tensor was passed
                input_sequence_list = [inputs[:, t, ...] for t in range(tf.shape(inputs)[1])]
            else:
                input_sequence_list = inputs

            input_sequence_tensor = tf.stack(input_sequence_list, axis=1) 
            skip_connection_sequences = self.encoder_model(input_sequence_tensor, training=training)
            all_time_step_outputs = []
            for t in range(tf.shape(input_sequence_tensor)[1]): 
                decoder_inputs_step_t = [skip_seq[:, t, ...] for skip_seq in skip_connection_sequences]
                decoder_inputs_ordered = [
                    decoder_inputs_step_t[5], decoder_inputs_step_t[4], decoder_inputs_step_t[3],
                    decoder_inputs_step_t[2], decoder_inputs_step_t[1], decoder_inputs_step_t[0]
                ]
                denoised_image_step = self.decoder_model(decoder_inputs_ordered, training=training)
                all_time_step_outputs.append(denoised_image_step)
            return all_time_step_outputs # Returns a list of tensors (one per time step)

        elif self.model_type_str == "AE":
            # Expects a single batch tensor [batch, H, W, C] for AE mode.
            # If a list is passed (e.g. from train_step), take the first element.
            input_batch_tensor = inputs[0] if isinstance(inputs, list) else inputs
            
            skip_connections_step = self.encoder_model(input_batch_tensor, training=training) 
            decoder_inputs_ordered = [
                skip_connections_step[5], skip_connections_step[4], skip_connections_step[3],
                skip_connections_step[2], skip_connections_step[1], skip_connections_step[0]
            ]
            denoised_image_batch = self.decoder_model(decoder_inputs_ordered, training=training)
            # For AE (seqLen=1), the output should also be a list containing one tensor,
            # to maintain consistency with how loss functions and image saving expect the output.
            return [denoised_image_batch] 
        
        return [] # Should not reach here

    @tf.function
    def train_step(self, data):
        # data is (inputNoise_batch_tensor_list, inputClean_batch_tensor_list)
        # For AE (seqLen=1), each list contains one tensor: ([noisy_batch], [clean_batch])
        inputNoise_data, inputClean_data = data 
        
        with tf.GradientTape() as tape:
            # self() calls the 'call' method.
            # For AE, inputNoise_data is [noisy_batch_tensor]. call will take noisy_batch_tensor.
            # For RAE, inputNoise_data is [step1_tensor, step2_tensor, ...]. call will take this list.
            denoised_img_pred_list = self(inputNoise_data, training=True) 
            
            # Loss functions expect lists of tensors (even if list has one element for AE)
            spatial_loss = self._get_L1_loss(denoised_img_pred_list, inputClean_data)
            gradient_loss = self._get_grad_L1_loss(denoised_img_pred_list, inputClean_data)
            
            current_total_loss = self.ws * spatial_loss + self.wg * gradient_loss
            loss_metrics = {'Spatial_loss': spatial_loss, 'Gradient_loss': gradient_loss}

            if self.model_type_str == 'RAE' and self.seqLen > 1: 
                temporal_loss = self._get_tem_L1_loss(denoised_img_pred_list, inputClean_data)
                current_total_loss += self.wt * temporal_loss
                loss_metrics['Temporal_loss'] = temporal_loss
            
            loss_metrics['Total_loss'] = current_total_loss
        
        model_vars = self.encoder_model.trainable_variables + self.decoder_model.trainable_variables
        if not model_vars:
            tf.print("警告: 未找到可訓練的變量。")
            return loss_metrics
        gradients = tape.gradient(current_total_loss, model_vars)
        valid_grads_and_vars = [(g, v) for g, v in zip(gradients, model_vars) if g is not None]
        if not valid_grads_and_vars:
            tf.print("警告: 未為任何可訓練變量計算梯度。")
            return loss_metrics
        self.optimizer.apply_gradients(valid_grads_and_vars)

        if self.phase == 'TRAIN' and self.train_writer is not None:
            with self.train_writer.as_default(step=self.optimizer.iterations): 
                for name, metric_val in loss_metrics.items():
                    tf.summary.scalar(name, metric_val)
        return loss_metrics

    # Loss functions now expect lists of tensors (outputList, targetList)
    # For AE mode (seqLen=1), these lists will contain a single batch tensor.
    def _preprocess(self, inp): return inp 
    def _get_L1_loss(self, outputList, targetList): # outputList is [denoised_batch_tensor] for AE
        if not outputList or not targetList: return tf.constant(0.0, dtype=tf.float32)
        out_p = self._preprocess(outputList[0]) # Takes the first (and only for AE) tensor from the list
        target_p = self._preprocess(targetList[0])
        return tf.reduce_mean(tf.abs(out_p - target_p))

    def _get_grad_L1_loss(self, outputList, targetList):
        if not outputList or not targetList: return tf.constant(0.0, dtype=tf.float32)
        out_grad = self._applyGrad(self._preprocess(outputList[0]))
        target_grad = self._applyGrad(self._preprocess(targetList[0]))
        return tf.reduce_mean(tf.abs(out_grad - target_grad))

    def _get_tem_L1_loss(self, outputList, targetList): # Only called if RAE and seqLen > 1
        if len(outputList) < 2 or len(targetList) < 2: return tf.constant(0.0, dtype=tf.float32)
        lossSum = tf.constant(0.0, dtype=tf.float32)
        outPrev, targetPrev = self._preprocess(outputList[0]), self._preprocess(targetList[0])
        for i in range(1, len(outputList)):
            out_curr,target_curr = self._preprocess(outputList[i]),self._preprocess(targetList[i])
            lossSum += tf.reduce_mean(tf.abs((out_curr-outPrev)-(target_curr-targetPrev)))
            outPrev,targetPrev = out_curr,target_curr
        return lossSum / tf.cast(len(outputList) - 1, tf.float32)

    def _applyGrad(self, inp): # inp is a batch tensor [batch, H, W, C]
        num_channels = tf.shape(inp)[-1]
        log_kernel_2d_np = self.LoG_filter()
        log_kernel_2d_tf = tf.convert_to_tensor(log_kernel_2d_np,dtype=tf.float32)
        log_kernel_4d_s = tf.reshape(log_kernel_2d_tf, [15,15,1,1])
        depthwise_kernel = tf.tile(log_kernel_4d_s, [1,1,num_channels,1])
        return tf.nn.depthwise_conv2d(inp, depthwise_kernel, strides=[1,1,1,1], padding="SAME", data_format='NHWC')
    def LoG_filter(self):
        nx,ny=(15,15);x=np.linspace(-7,7,nx,dtype=np.float32);y=np.linspace(-7,7,ny,dtype=np.float32)
        xv,yv=np.meshgrid(x,y);return self.LoG(xv,yv,1.5)
    def LoG(self,X,Y,sig):return -1.0/(math.pi*sig**4.0)*(1.0-(X**2+Y**2)/(2.0*sig**2))*np.exp(-(X**2+Y**2)/(2.0*sig**2))
    
    def _normalize_for_saving_and_tensorboard(self, image_np_or_tensor):
        if isinstance(image_np_or_tensor, tf.Tensor): image_np = image_np_or_tensor.numpy() 
        else: image_np = image_np_or_tensor
        if np.issubdtype(image_np.dtype, np.floating): img = np.add(np.multiply(image_np, 128.), 128.)
        else: img = image_np
        return np.clip(img, 0, 255).astype(np.uint8)

    def save_comparison_images(self, 
                               input_noise_batch_normalized, # Direct batch tensor [batch, H, W, C]
                               original_noise_batch_uint8_rgb,
                               original_clean_batch_uint8_rgb,
                               directory, epoch, global_iter_idx, current_phase_str):
        imgs_folder = os.path.join(directory, f'epoch{epoch}_iter{global_iter_idx}_{current_phase_str}')
        os.makedirs(imgs_folder, exist_ok=True)

        # For AE (seqLen=1), self() expects a list containing one tensor.
        denoised_batch_tensor_list = self([input_noise_batch_normalized], training=False)
        denoised_batch_tensor = denoised_batch_tensor_list[0] 
        batch_size = tf.shape(denoised_batch_tensor)[0].numpy()

        for i in range(batch_size): 
            noisy_to_save_uint8 = original_noise_batch_uint8_rgb[i] 
            denoised_to_save_uint8 = self._normalize_for_saving_and_tensorboard(denoised_batch_tensor[i])
            clean_to_save_uint8 = original_clean_batch_uint8_rgb[i]
            file_prefix = f"datasample{global_iter_idx}" 
            if batch_size > 1: file_prefix += f"_batchidx{i}"
            try:
                #imageio.imwrite(os.path.join(imgs_folder, f'{file_prefix}_input_noisy.png'), noisy_to_save_uint8)
                imageio.imwrite(os.path.join(imgs_folder, f'{file_prefix}_output_denoised.png'), denoised_to_save_uint8)
                #imageio.imwrite(os.path.join(imgs_folder, f'{file_prefix}_target_clean.png'), clean_to_save_uint8)
                if self.phase == 'TRAIN' and self.train_writer is not None and i == 0: 
                    with self.train_writer.as_default(step=self.optimizer.iterations):
                        #tf.summary.image(f"TrainingEpoch{epoch}/InputNoisy_Iter{global_iter_idx}", tf.expand_dims(noisy_to_save_uint8, 0), max_outputs=1)
                        tf.summary.image(f"TrainingEpoch{epoch}/OutputDenoised_Iter{global_iter_idx}", tf.expand_dims(denoised_to_save_uint8, 0), max_outputs=1)
                        #tf.summary.image(f"TrainingEpoch{epoch}/TargetClean_Iter{global_iter_idx}", tf.expand_dims(clean_to_save_uint8, 0), max_outputs=1)
            except Exception as e: print(f"保存圖像或寫入TensorBoard時出錯 (iter {global_iter_idx}, batch_idx {i}): {e}")
        if batch_size > 0: print(f"已將比較圖像保存到 {imgs_folder} (全局迭代 {global_iter_idx})")