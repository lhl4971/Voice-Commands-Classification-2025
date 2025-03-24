## Voice Commands Classification 2025

Eugene Ilyushin. Voice Commands Classification 2025. https://kaggle.com/competitions/voice-commands-classification-2025, 2025. Kaggle.

In this competition, we build a neural network to classify audio samples with a duration of 1 second and a sampling rate of 16000 Hz.

The final accuracy achieved is as follows:
- Training set: 99.99%
- Validation set: 95.12%
- Test set: 89.14%

We use MFCC to extract frequency features from the original waveform and train the waveform encoder and spectrum encoder separately.  
The waveform encoder samples 10ms waveform data into 1 frame and uses a 2-layer bidirectional LSTM network with a residual structure.  
The spectral encoder uses a 3-layer self-attention encoder with Rotary Positional Encoding, where the hidden layer size is 120.
The waveform encoder and spectral encoder are trained separately.

We then apply a cross-attention mechanism to the waveform information and the spectral information, where the spectral information is used as Q and the waveform information is used as KV, and fine-tune the model on the given dataset.