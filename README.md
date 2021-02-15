# VAE
```python
"""
  Random Normal latent에 대한 결과를 출력해 보았다 N(0,1)
"""
z = torch.randn(size = (1,128)).to(DEVICE)
pred = vae.decoder(z)
image = pred[0].view(28,28)
plt.imshow(image.cpu().detach().numpy())

```

20epoch 훈련
![image](https://github.com/yhy258/VAE/blob/master/vaeresult.png?raw=true)
