<h1 align="center">Random  Anime Face Generation with Generative Adversarial Networks</h1>

[Training Dataset](https://www.kaggle.com/splcher/animefacedataset)<br/>
[DCGAN PyTorch Tutorial](https://youtu.be/IZtv9s_Wx9I)<br/>
[WGAN PyTorch Tutorial](https://youtu.be/pG0QZ7OddX4)

## Data Sample
<img src="https://github.com/ANI717/WGAN-GP_Anime_Face_Generator/blob/main/result/real.png" alt="real_image" class="inline"/><br/>

## Generated Anime Faces with DCGAN
`Epochs = 10`<br/>
`Batch Size = 128`<br/>
<img src="https://github.com/ANI717/Anime_Face_Generation_GANs/blob/main/DCGAN/result/fake_100_64.png" alt="fake_image_1" class="inline"/><br/>
<br/>
When Feature Size is Doubled for both Generator and Descriminator.<br/>
<img src="https://github.com/ANI717/Anime_Face_Generation_GANs/blob/main/DCGAN/result/fake_200_128.png" alt="fake_image_1" class="inline"/><br/>

## Generated Anime Faces with WGAN
Gradiend Penalty is Applied while Training.<br/>
`Epochs = 10`<br/>
`Batch Size = 128`<br/>
<img src="https://github.com/ANI717/Anime_Face_Generation_GANs/blob/main/WGAN-GP/result/fake.png" alt="fake_image_1" class="inline"/><br/>
