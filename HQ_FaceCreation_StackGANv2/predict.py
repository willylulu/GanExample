from keras.models import load_model

gnet = load_model("stackGan_v2-v3.h5")

noise = np.random.normal(0, 1.0, size=[16, 100])
fake_64, fake_128, fake_256 = gnet.predict(noise)
saveImages('./', fake_64, 64, 0)
saveImages('./', fake_128, 128, 0)
saveImages('./', fake_256, 256, 0)