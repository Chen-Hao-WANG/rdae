import glob
import numpy as np
import os
import os.path as osp
from PIL import Image 
import random
from tqdm import tqdm

class BatchLoader(object):
    def __init__(self, dataRoot, inpChannel = 3, batchSize = 32, seqLen = 1, 
                 scaleSize = (425, 800), cropSize=(128, 128),
                 isRandom=True, phase='TRAIN', rseed = None, process_original_size=False):
        
        self.dataRoot = dataRoot
        self.batchSize = batchSize
        self.seqLen = seqLen # For AE model, this will effectively be 1
        self.inpChannel = inpChannel 
        self.process_original_size = process_original_size

        if self.process_original_size:
            self.model_inputH = cropSize[0]
            self.model_inputW = cropSize[1]
            print(f"數據加載器配置為加載原始尺寸圖像，期望模型輸入: {self.model_inputH}x{self.model_inputW}")
        else:
            self.scaleH = scaleSize[0]
            self.scaleW = scaleSize[1]
            self.cropH = cropSize[0]
            self.cropW = cropSize[1]
            assert(self.cropH <= self.scaleH), "裁剪高度必須小於或等於縮放高度。"
            assert(self.cropW <= self.scaleW), "裁剪寬度必須小於或等於縮放寬度。"
            self.model_inputH = self.cropH
            self.model_inputW = self.cropW

        self.phase = phase.upper()
        
        if rseed is not None:
            random.seed(rseed)
            np.random.seed(rseed)

        self.imageListNoise, self.imageListClean = self.create_image_denoising_splits(dataRoot, phase.lower())
        self.count = len(self.imageListNoise) 
        
        if self.count == 0:
            clean_target_name_for_error = self.get_clean_target_name()
            raise FileNotFoundError(
                f"在 {dataRoot} 中未找到適合 {self.phase} 階段的圖像對。 "
                f"期望 '{clean_target_name_for_error}' 作為乾淨目標，並至少有一個 .ppm 檔案作為噪聲輸入。"
            )
        
        print(f"為 {self.phase} 階段找到 {self.count} 個噪聲/乾淨圖像對。")
        if self.count < self.batchSize and self.phase == 'TRAIN':
            print(f"警告: 數據集大小 ({self.count}) 小於批次大小 ({self.batchSize})。訓練時樣本將被重複使用。")

        self.perm = list(range(self.count))
        if isRandom and self.phase == 'TRAIN':
            random.shuffle(self.perm)
        self.cur = 0      

    def get_clean_target_name(self):
        return "result_25k.png"

    def create_image_denoising_splits(self, data_root_dir, phase_lower):
        print(f"正在掃描 {data_root_dir} 中的圖像檔案 (階段: {phase_lower})")
        clean_target_name = self.get_clean_target_name()
        clean_target_path = osp.join(data_root_dir, clean_target_name)
        imageListNoise_seqs, imageListClean_seqs = [], []
        if not osp.isfile(clean_target_path):
            print(f"錯誤: 在 '{data_root_dir}' 中未找到乾淨目標圖像 '{clean_target_name}'。請確保該檔案存在。")
            return [], []
        noisy_ppm_files = glob.glob(osp.join(data_root_dir, "*.ppm"))
        if not noisy_ppm_files:
            print(f"警告: 在 '{data_root_dir}' 中未找到任何 .ppm 檔案作為噪聲輸入。")
            return [], []
        for noisy_path in noisy_ppm_files:
            if osp.isfile(noisy_path):
                imageListNoise_seqs.append([noisy_path]) 
                imageListClean_seqs.append([clean_target_path])
            else:
                print(f"警告: 掃描到的噪聲輸入圖像 '{noisy_path}' 似乎不存在。")
        if not imageListNoise_seqs:
            print(f"警告: 未能在 '{data_root_dir}' 中找到任何可用的噪聲輸入 .ppm 檔案。")
        return imageListNoise_seqs, imageListClean_seqs

    def loadBatch(self):
        isNewEpoch = False
        # Initialize batch arrays for the full self.batchSize
        batch_inputNoise_norm = np.zeros((self.batchSize, self.model_inputH, self.model_inputW, self.inpChannel), dtype=np.float32)
        batch_inputClean_norm = np.zeros((self.batchSize, self.model_inputH, self.model_inputW, 3), dtype=np.float32) 
        batch_inputNoise_orig_rgb = np.zeros((self.batchSize, self.model_inputH, self.model_inputW, 3), dtype=np.uint8)
        batch_inputClean_orig_rgb = np.zeros((self.batchSize, self.model_inputH, self.model_inputW, 3), dtype=np.uint8)

        for i in range(self.batchSize):
            if self.cur >= self.count: 
                self.cur = 0 
                if self.phase == 'TRAIN': random.shuffle(self.perm) 
                isNewEpoch = True 
            idx_in_perm = self.perm[self.cur % self.count] 
            # Since seqLen is 1, imageListNoise[idx_in_perm] is like ['path/to/img.ppm']
            noise_img_path = self.imageListNoise[idx_in_perm][0] 
            clean_img_path = self.imageListClean[idx_in_perm][0] 
            paths_to_load_dict = {'noise': [noise_img_path], 'clean': [clean_img_path]}
            loaded_data_dict = self.loadSingleImagePair(paths_to_load_dict)

            if loaded_data_dict['noise_norm'] is not None and loaded_data_dict['clean_norm'] is not None:
                batch_inputNoise_norm[i, ...] = loaded_data_dict['noise_norm']
                batch_inputClean_norm[i, ...] = loaded_data_dict['clean_norm']
                
                noise_orig_data = loaded_data_dict['noise_orig'] 
                if noise_orig_data.shape[-1] == 3:
                    batch_inputNoise_orig_rgb[i, ...] = noise_orig_data
                elif noise_orig_data.shape[-1] == 1: 
                    batch_inputNoise_orig_rgb[i, ...] = np.concatenate([noise_orig_data]*3, axis=-1)
                elif noise_orig_data.shape[-1] > 3: 
                    batch_inputNoise_orig_rgb[i, ...] = noise_orig_data[..., :3] 
                else: 
                     batch_inputNoise_orig_rgb[i, ...] = np.zeros((self.model_inputH, self.model_inputW, 3), dtype=np.uint8)
                batch_inputClean_orig_rgb[i, ...] = loaded_data_dict['clean_orig']
            self.cur += 1 
        
        # For seqLen=1 (static image denoising), return the batch tensors directly,
        # not as lists containing a single tensor.
        # main.py will wrap them in a list if RAE.call expects a list.
        if self.cur == 0 and self.phase == 'TRAIN' and not isNewEpoch : isNewEpoch = True
        return batch_inputNoise_norm, batch_inputNoise_orig_rgb, batch_inputClean_norm, batch_inputClean_orig_rgb, isNewEpoch

    def loadSingleImagePair(self, paths_dict_single_seq):
        results = {'noise_norm': None, 'noise_orig': None, 'clean_norm': None, 'clean_orig': None}
        
        for key, path_list_single_item in paths_dict_single_seq.items():
            img_path = path_list_single_item[0] 
            try:
                img = Image.open(img_path)
                target_channels_for_key = self.inpChannel if key == 'noise' else 3
                target_mode_for_key = 'RGB' if target_channels_for_key == 3 else 'L'

                if img.mode != target_mode_for_key:
                    img = img.convert(target_mode_for_key)

                if self.process_original_size:
                    if img.height != self.model_inputH or img.width != self.model_inputW:
                        img = img.resize((self.model_inputW, self.model_inputH), Image.Resampling.LANCZOS)
                    img_np_final_shape = np.asarray(img, dtype=np.float32)
                    if img_np_final_shape.ndim == 2: img_np_final_shape = np.expand_dims(img_np_final_shape, axis=-1)
                else: 
                    img = img.resize((self.scaleW, self.scaleH), Image.Resampling.LANCZOS)
                    img_np_for_crop = np.asarray(img, dtype=np.float32) 
                    if img_np_for_crop.ndim == 2: img_np_for_crop = np.expand_dims(img_np_for_crop, axis=-1)
                    if img_np_for_crop.shape[-1] != target_channels_for_key:
                        if target_channels_for_key == 1 and img_np_for_crop.shape[-1] != 1:
                            img_np_for_crop = img_np_for_crop[:,:,0:1] 
                        elif target_channels_for_key == 3:
                            if img_np_for_crop.shape[-1] == 1: img_np_for_crop = np.concatenate([img_np_for_crop]*3, axis=-1)
                            elif img_np_for_crop.shape[-1] == 4: img_np_for_crop = img_np_for_crop[:,:,:3]
                    max_h_offset = img_np_for_crop.shape[0] - self.cropH 
                    max_w_offset = img_np_for_crop.shape[1] - self.cropW 
                    pointH = random.randrange(max_h_offset + 1) if max_h_offset >=0 else 0
                    pointW = random.randrange(max_w_offset + 1) if max_w_offset >=0 else 0
                    img_np_final_shape = img_np_for_crop[pointH : pointH + self.cropH, pointW : pointW + self.cropW, :]
            except FileNotFoundError:
                print(f"錯誤: 圖像文件未找到於 {img_path}")
                return results 
            except Exception as e:
                print(f"錯誤: 打開或處理圖像 {img_path} 時出錯: {e}")
                return results

            if img_np_final_shape.shape[-1] != target_channels_for_key:
                if target_channels_for_key == 1 and img_np_final_shape.shape[-1] > 1: 
                    img_np_final_shape = img_np_final_shape[:,:,0:1] 
                elif target_channels_for_key == 3:
                    if img_np_final_shape.shape[-1] == 1: img_np_final_shape = np.concatenate([img_np_final_shape]*3, axis=-1)
                    elif img_np_final_shape.shape[-1] == 4: img_np_final_shape = img_np_final_shape[:,:,:3]
                    elif img_np_final_shape.shape[-1] != 3 : 
                        if img_np_final_shape.shape[-1] > 3: img_np_final_shape = img_np_final_shape[:,:,:3]
                        else: 
                            padding = np.zeros((img_np_final_shape.shape[0], img_np_final_shape.shape[1], 3 - img_np_final_shape.shape[-1]), dtype=img_np_final_shape.dtype)
                            img_np_final_shape = np.concatenate([img_np_final_shape, padding], axis=-1)
            
            original_img_to_store = np.clip(img_np_final_shape, 0, 255).astype(np.uint8)
            normalized_img = (img_np_final_shape.astype(np.float32) - 128.0) / 128.0
            
            if key == 'noise':
                results['noise_norm'] = normalized_img
                results['noise_orig'] = original_img_to_store 
            elif key == 'clean':
                results['clean_norm'] = normalized_img
                results['clean_orig'] = original_img_to_store
        return results
