# MAT-UNet
This is a personal practice project. The model exhibited severe overfitting due to the insufficient size of the dataset relative to the model capacity.

- Dataset: Kvasir-SEG
- GPU: A6000
- Training Time: 4.5h
- Best Validation Dice: 84%

## Model Architecture
- three-layer Convolutional Variational Autoencoder (VAE)
- using **Squeeze-and-Excitation (SE)** modules at skip connections
- Integrating **Multi-Head Cross Attention (MHCA)** in the deeper layers of the decoder, combining both **Channel Attention** and **Spatial Attention**
- Loss = 0.5 * BCE + 0.5 * Dice

## Results
<table>
  <tr>
    <td width="33%"><img src="/asserts/train_tatal_loss.png"></td>
    <td width="33%"><img src="/asserts/train_dice_loss.png"></td>
    <td width="33%"><img src="/asserts/train_bce_loss.png"></td>
  </tr>
  <tr>
    <td width="33%"><img src="/asserts/val_total_loss.png"></td>
    <td width="33%"><img src="/asserts/val_dice_loss.png"></td>
    <td width="33%"><img src="/asserts/val_bce_loss.png"></td>
  </tr>
</table>
