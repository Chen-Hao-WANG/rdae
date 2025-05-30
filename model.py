import os
import imageio 
import numpy as np
from tqdm import tqdm 
import time

class Model(object):
    """
    Base Model class (Legacy).
    In TensorFlow 2.x with Keras, specific model logic (forward pass, training step)
    is typically implemented by subclassing tf.keras.Model.
    This class is kept for compatibility if RAE was inheriting from it,
    but RAE should now inherit directly from tf.keras.Model.
    """

    def __init__(self, *args, **kwargs):
        # Basic initialization, can be extended by subclasses
        # print("Warning: Legacy Model base class initialized. Consider refactoring to directly use tf.keras.Model.")
        pass

    # This method is effectively replaced by the train_step in a Keras model
    # or by using model.compile() and model.fit().
    def update_params(self, input_tensor):
        '''Update parameters of the network (Legacy method)'''
        raise NotImplementedError("This method should be implemented in the Keras model subclass as 'train_step' or handled by 'fit'.")

    # Image generation and utility methods (_toThreeC_np, _compact_batch_img_np)
    # have been moved to the RAE class (rae.py) as they are more closely
    # tied to its prediction output and specific visualization needs.
    # If there were truly generic utilities, they could be static methods here.
    
    # Example of how they might look if they were static here (for reference):
    @staticmethod
    def _toThreeC_np_static(a):
        if a.ndim == 2: 
            return np.stack([a,a,a], axis=-1)
        elif a.ndim == 3 and a.shape[-1] == 1:
             return np.concatenate([a,a,a], axis=-1)
        elif a.ndim == 3 and a.shape[-1] == 3: # Already 3 channels
            return a
        else: 
            # print(f"Warning: toThreeC_np_static received unexpected shape {a.shape}.")
            # Fallback: try to take first channel if multi-channel but not 1 or 3
            if a.ndim == 3 and a.shape[-1] > 1:
                return np.stack([a[...,0]]*3, axis=-1) 
            return a # Return as is if still problematic

    @staticmethod
    def _compact_batch_img_np_static(input_nparyList, num_to_show_in_row):
        rowList = []
        if not input_nparyList: return np.zeros((100,100,3), dtype=np.uint8)

        # Determine a common height for all images in a row for proper hstack
        # This is a simplification; more robust resizing/padding might be needed
        # if source images in input_nparyList have very different aspect ratios/heights.
        # Let's try to find a target height from the first image of the first source.
        target_height = -1
        if input_nparyList[0].shape[0] > 0 : # Check if first source has samples
            if input_nparyList[0][0].ndim >=2 : # Check if first sample is an image
                 target_height = input_nparyList[0][0].shape[0]
        
        if target_height <=0 : target_height = 128 # Default fallback height

        for row_data_source in input_nparyList:
            if row_data_source.shape[0] == 0: continue # Skip if no images in this source

            actual_num_in_row = min(num_to_show_in_row, row_data_source.shape[0])
            if actual_num_in_row == 0: continue

            compact_row_parts = []
            for i in range(actual_num_in_row):
                img_part = row_data_source[i]
                # Ensure 3 channels and consistent height
                img_part_3c = Model._toThreeC_np_static(np.squeeze(img_part))
                
                if img_part_3c.shape[0] != target_height:
                    # Basic resize - might distort aspect ratio.
                    # Consider padding or more sophisticated resizing if aspect ratio preservation is key.
                    new_width = int(img_part_3c.shape[1] * (target_height / img_part_3c.shape[0]))
                    if new_width <=0 : new_width = target_height # fallback width
                    try:
                        pil_img = Image.fromarray(img_part_3c.astype(np.uint8))
                        resized_pil_img = pil_img.resize((new_width, target_height), Image.Resampling.LANCZOS)
                        img_part_3c = np.array(resized_pil_img)
                    except Exception as e:
                        # print(f"Error resizing image part: {e}. Skipping part.")
                        continue # Skip this part if resizing fails

                if img_part_3c.ndim != 3 or img_part_3c.shape[-1] != 3: continue 
                compact_row_parts.append(img_part_3c)
            
            if compact_row_parts:
                try:
                    rowList.append(np.hstack(compact_row_parts))
                except ValueError as ve:
                    # print(f"ValueError during hstack (likely inconsistent heights): {ve}")
                    # Fallback: skip this row or add placeholder
                    pass # For now, skip problematic rows
        
        if not rowList: return np.zeros((target_height if target_height > 0 else 100, 100,3), dtype=np.uint8)
        
        max_width = 0
        for r_final in rowList: max_width = max(max_width, r_final.shape[1])
        
        padded_final_rows = []
        for r_final in rowList:
            pad_amount = max_width - r_final.shape[1]
            if pad_amount > 0:
                padding_array = np.zeros((r_final.shape[0], pad_amount, r_final.shape[2]), dtype=r_final.dtype)
                padded_final_rows.append(np.hstack([r_final, padding_array]))
            else:
                padded_final_rows.append(r_final)

        if not padded_final_rows: return np.zeros((target_height if target_height > 0 else 100, 100,3), dtype=np.uint8)
        return np.vstack(padded_final_rows)

