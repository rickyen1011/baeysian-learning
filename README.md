# quantum-ml-main
Webpage source files for tutorial at IJCAI 201

https://huckiyang.github.io/quantum-ml-main/

```python
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import models
# input shape (None, 60, 126, 1)
cnn_model = keras.models.load_model('/content/QuantumSpeech-QCNN/checkpoints/0910_1843_conv_sp2cmd.hdf5')
if mel_feat.shape[1] != 126:
    print("Mel Feat before Cutting", mel_feat.shape)
    test_feat = mel_feat[:, 0:126]
else:
    test_feat = mel_feat

test_feat = np.expand_dims(test_feat, axis=(0,3))
test_label = cnn_model.predict(test_feat)
labels = [
    'left', 'go', 'yes', 'down', 'up', 'on', 'right', 'no', 'off', 'stop',
]

print("prediction:", labels[np.argmax(test_label)])
```
