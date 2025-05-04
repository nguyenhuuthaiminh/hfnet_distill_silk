# HFNet Distill Silk

This project is a distilled version of [HF-Net](https://github.com/ethz-asl/hfnet), designed for lightweight deployment and efficient inference. Unlike the original, the teacher model used in this approach is [SiLK](https://github.com/facebookresearch/silk), replacing SuperPoint from the original paper.

---

## 🚀 Target

The main goal of this project is to train a distilled version of HF-Net using SiLK as the teacher model. SiLK has demonstrated strong performance in both keypoint detection and local descriptor extraction, which are critical components in visual SLAM (Simultaneous Localization and Mapping) systems. By leveraging knowledge distillation, we aim to transfer the strengths of SiLK into a smaller and more efficient student network based on HF-Net. This approach allows for lightweight deployment on resource-constrained platforms, such as mobile devices or embedded systems, without significantly compromising accuracy. The resulting model maintains reliable feature extraction capabilities while offering faster inference times, making it suitable for real-time SLAM applications.

## ⭐ Milestones

1. Verification of the Results from the Original Paper
2. Reproducing HF-Net in Pytorch and Test the model
3. Exploring Multiple Approaches to Improve HF-Net Performance
4. Training the model with a private dataset

## 1. Verification of the Results from the Original Paper

In the first phase, I executed all the provided inference codes to validate the published results of HF-Net and SiLK using the HPatches dataset. This involved running experiments to reproduce the reported performance metrics on this dataset, analyzing the outcomes, and ensuring consistency with the original findings. By focusing on HPatches, I established a solid foundation for further development and enhancements of the models.

## 2. Reproducing HF-Net in Pytorch and Test the model

In the second step, I reimplemented HF-Net in PyTorch and trained the model [hfnet_superpoint](https://www.kaggle.com/code/moletet/hfnet-superpoint) using 1,000 images from the Google Landmarks dataset in 19 epoch. This step ensured the correctness of the implementation and evaluated the model's performance on a diverse and challenging dataset. 

After that, I used the pre-trained model (hfnet-superpoint(SDS)) to test it with my evaluation module on the HPatches dataset. This allowed me to assess the model's performance and validate its effectiveness in a real-world evaluation scenario. And here are the outcomes I obtained:

**Note**: My model can be easily trained locally. You only need to configure it to train with a GPU and prepare the data in the correct format for the three heads (local detector, local descriptor, and global descriptor). Add 4 paths (`IMAGE_DIR,GLOBAL_DIR,LOCAL_DIR,SAVE_DIR`) in [config.py](model/training/config.py) and run this line of code in terminal:
```bash
python -m model.hfnet_distill_silk
```

### Configuration

The output from my reproduced model in y,x order and to evaluation with belong metric I should flip the output to adapt with the evaluation module (OpenCV).

```python
eval_config = {
   'num features': 300,
   'do_ratio_test': True,
   'correct_match_thresh': 3,
   'correct_H_thresh': 3,
}
```

| Model                      | MS    | Homography | mAP (Desc) | MLE  | Rep  | mAP (Det) |
|----------------------------|-------|------------|------------|------|------|-----------|
| superpoint-pretrain        | 0.441 | 0.810      | 0.846      | 1.036| 0.495| 0.276     |
| hfnet-pretrain (ckpt 83096)| 0.463 | 0.802      | 0.844      | 1.093| 0.519| 0.327     |
| hfnet-superpoint (SDS)     | 0.309 | 0.643      | 0.761      | 1.302| 0.365| 0.181     |

**Note**
- **MS**: Percentage of correct matches under a distance threshold
- **Homography**: Accuracy of the estimated homography transformation.
- **mAP (Desc)**: Mean Average Precision of the local descriptor on matching tasks.
- **MLE**: Mean Localization Error — average distance between predicted and ground-truth keypoints.
- **Rep (Repeatability)**: Ratio of repeatable keypoints between two views.
- **mAP (Det)**: Mean Average Precision of the detector under varying thresholds.

- **superpoint-pretrain**: Pre-trained SuperPoint teacher model.
- **hfnet-pretrain (ckpt 83096)**: HF-Net distilled from the SuperPoint model using the original paper's weights.
- **hfnet-superpoint (SDS)**: HF-Net reproduced in PyTorch, distilled from SuperPoint using a small dataset (SDS).

As you can see, the results of the hfnet-superpoint (SDS) are likely to improve if the model is trained with a larger dataset and for a longer duration.

## 3. Exploring Multiple Approaches to Improve HF-Net Performance

### 1️⃣ Attempt 1

In this attempt, I used `sparse_positions` and `sparse_descriptors`, which contain the positions and descriptors of each keypoint. I converted the `sparse_positions` into a `heatmap` to adapt it to the loss function of this model, which worked very well during the training process. Additionally, to address the difference of `sparse_descriptors` from SiLK and `local_descriptor_map` from HF_Net, I implemented a post-processing module to extract the output of HF-Net. This allowed me to align the keypoint positions and apply the MSE loss to this head effectively. However, when I evaluated the model on the HPatches dataset using the same configuration as above, I obtained the following result:

| Model                            | MS    | Homography | mAP (Desc) | MLE  | Rep  | mAP (Det) |
|----------------------------------|-------|------------|------------|------|------|-----------|
| hfnet-superpoint (SDS)           | 0.309 | 0.643      | 0.761      | 1.302| 0.365| 0.181     |
| hfnet-silk (sparse_position SDS) | 0.045 | 0.132      | 0.092      | 1.411| 0.227| --------- |

As you can see, the results are not satisfactory enough to proceed to the next step.

### 2️⃣ Attempt 2 

In the second scenario, I tried another approach by using `raw_descriptors` (similar to `local_descriptor_map`) instead of `sparse_descriptors` from SiLK. The `raw_descriptors` have a shape of (128, H-18, W-18). 

To align the output of HF-Net with the teacher model's output, I upscaled the HF-Net output using the `grid_sample` function in PyTorch. Below is an example of how this was implemented:

| Model                            | MS    | Homography | mAP (Desc) | MLE  | Rep  | mAP (Det) |
|----------------------------------|-------|------------|------------|------|------|-----------|
| hfnet-superpoint (SDS)           | 0.309 | 0.643      | 0.761      | 1.302| 0.365| 0.181     |
| hfnet(up)-silk (raw_descriptors) | 0.051 | 0.071      | 0.098      | 1.665| 0.235| 0.077     |

The low performance of `hfnet-silk (raw_descriptors SDS)` could be attributed to the misalignment between the teacher model's `raw_descriptors` and the HF-Net output.

### 3️⃣ Attempt 3

In the most recent trial of distilling SiLK for HF-Net, I attempted to align SiLK's `raw_descriptors` to the same shape as the output of the `local_descriptor_map` from HF-Net (128, H/8, W/8). The results show a noticeable improvement compared to the first two attempts but are still not as good as expected.

| Model                              | MS    | Homography | mAP (Desc) | MLE  | Rep  | mAP (Det) |
|------------------------------------|-------|------------|------------|------|------|-----------|
| hfnet-superpoint (SDS)             | 0.309 | 0.643      | 0.761      | 1.302| 0.365| 0.181     |
| hfnet-silk(down) (raw_descriptors) | 0.201 | 0.347      | 0.415      | 1.504| 0.375| 0.192     |

As you can see, some metrics (`Rep`and `mAp`) of the detector `hfnet-silk` are slightly better than those of `hfnet-superpoint`. Therefore, I would like to visualize some results when using it to detect keypoints on the HPatches dataset.

![HF-Net(SiLK)](images/hfnet-silk.png "HF-Net(SiLK) Keypoint Detection on HPatches")
*Figure 1: Keypoint detection results using HF-Net distilled from SiLK.*
![HF-Net(SuperPoint)](images/hfnet-superpoint.png "HF-Net(SuperPoint) Keypoint Detection on HPatches")
*Figure 2: Keypoint detection results using HF-Net distilled from SuperPoint.*

**Note**: In the same configuration, some result from `hfnet-silk` show really bad performance