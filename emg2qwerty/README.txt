ECE C147/C247 Final Project
Predicting Keystrokes from Surface EMG Signals
README

Summary
This submission contains the code and configuration files used for our final project on keystroke decoding from surface EMG (sEMG) using the emg2qwerty dataset. We compare the provided TDS convolutional baseline against recurrent and hybrid CNN+RNN architectures on the single-subject split for user #89335547. All models are trained with CTC loss and evaluated using Character Error Rate (CER).

Main reported results
- TDS Conv (baseline): Train CER 22.05, Val CER 22.62, Test CER 23.80
- Bidirectional GRU: Train CER 12.80, Val CER 17.10, Test CER 17.33
- CNN + BiLSTM: Train CER 4.22, Val CER 13.98, Test CER 14.96
- CNN + BiGRU: Train CER 6.08, Val CER 15.97, Test CER 15.32

Main conclusion
The TDS baseline is substantially improved by adding explicit bidirectional sequence modeling. The Bidirectional GRU reduces test CER from 23.80 to 17.33, showing that long-range temporal context is important for this task. Adding a temporal convolutional stage before the recurrent encoder improves performance further. CNN+BiLSTM gives the best overall result at 14.96 test CER, and CNN+BiGRU closely follows at 15.32, indicating that the hybrid CNN+RNN structure is especially effective for offline keystroke decoding from sEMG.

Files included in this submission
This submission includes code files, config files, and project write-up materials only, following the TA instruction not to submit checkpoints or logs.

Important modified files
- emg2qwerty/lightning.py
- emg2qwerty/modules.py
- config/model/rnn_ctc.yaml
- config/model/cnn_bigru_ctc.yaml
- config/model/cnn_bilstm_ctc.yaml
- emg2qwerty/train.py 
- main.tex / report PDF

Reported model ownership
- Kiko Trevino: Bidirectional GRU
- Sherine Chally: CNN + BiLSTM
- Anik Malik: CNN + BiGRU
- Janani Venkatramani: TDS baseline and exploratory vision-transformer-style model

Exploratory model
We also explored a Transformer / vision-transformer-style model as an architectural idea. It was treated as exploratory and was not included in the main quantitative comparison because its training behavior was unstable and its results were not competitive with the main recurrent and hybrid models.

Exact command used for the final CNN+BiGRU run
python -m emg2qwerty.train \
  model=cnn_bigru_ctc \
  user=single_user \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=50 \
  optimizer.lr=7e-4 \
  +trainer.gradient_clip_val=1.0

Typical run command for the Bidirectional GRU model
python -m emg2qwerty.train \
  model=rnn_ctc \
  user=single_user \
  trainer.accelerator=gpu \
  trainer.devices=1

Typical run command for the TDS baseline
python -m emg2qwerty.train \
  model=tds_conv_ctc \
  user=single_user \
  trainer.accelerator=gpu \
  trainer.devices=1

Exact command used for the CNN+BiLSTM model
python -m emg2qwerty.train \
  model=cnn_bilstm_ctc \
  user=single_user \
  trainer.accelerator=gpu \
  trainer.devices=1

Exact command used for the Transformer model
python -m emg2qwerty.train \
  model=vit \
  user=single_user \
  trainer.accelerator=gpu \
  trainer.devices=1

Alternatively, the CNN+BiLSTM model can be trained using the provided Jupyter notebook:
  cnn_bilstm_train.ipynb
Open the notebook and run all cells. It includes the full training pipeline and can be run in Google Colab or any local Jupyter environment with GPU support.

Notes on the configs
- rnn_ctc.yaml contains the Bidirectional GRU configuration.
- cnn_bigru_ctc.yaml contains the final CNN+BiGRU configuration used for the reported result.
- cnn_bilstm_ctc.yaml contains the CNN+BiLSTM configuration. The model architecture (cnn_channels, lstm_hidden_size, num_lstm_layers, dropout) is fully defined in that file and does not require any command-line overrides to reproduce the reported result.
- The reported CNN+BiGRU result used the exact command shown above, including the learning-rate override and gradient clipping.

Checkpoint note
The best model checkpoint was saved during training and retained separately for team reference, but is not included in the submission in accordance with the TA instruction not to submit checkpoints or logs.

Reproducibility note
The submitted code and configs are sufficient to reproduce the model architectures and training setup. We do not include checkpoints, logs, or large artifacts in this submission.
