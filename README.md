# Multilingual Brain Decoding via Unified Semantic Space

This repository contains the official PyTorch implementation for the paper: **"Multilingual Brain Decoding From Non-invasive Recordings via A Unified Semantic Space"**

## Abstract

Brain-computer interfaces (BCIs) with speech decoding from brain recordings have broad application potential in fields such as clinical rehabilitation and cognitive neuroscience. However, current decoding methods remain limited to single-language, single-subject, and single neuroimaging modality settings, restricting their clinical applicability and generalizability. Here we propose a joint multilingual, multi-subject and multimodal decoding framework. It maps diverse brain recordings into a unified semantic space defined by pre-trained multilingual models (PMMs), enabling decoding across multiple languages, multiple subjects and multiple neuroimaging modalities. The proposed framework is validated using non-invasive brain recordings from 159 participants across four languages. Results confirm the framework's feasibility across multilingual, multi-subject, and multimodal settings. More importantly, the proposed framework has the potential to facilitate linguistic fairness by enabling cross-lingual mapping within a unified semantic space, improving decoding performance for underrepresented languages in BCI applications. Overall, the proposed framework offers a potential paradigm for brain decoding in BCIs.

## Environment

- **OS**: Ubuntu 24.04.1 LTS
- **GPU**: NVIDIA A6000
- **Python**: 3.8.18
- **PyTorch**: 2.4.0
- **CUDA**: cu12

## Requirements

All required dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Project Structure

### `language_decoding/`
Contains ready-to-run test cases for evaluating the framework:
- **CN_decoding/**: Test case for Chinese (Mandarin) using SMN4Lang dataset
- **EN_decoding/**: Test case for English using LPPC-fMRI dataset

Each test directory includes:
- `get_Uniemb.py`: Maps brain activity recordings to the unified semantic space
- `gen_text_main.py`: Reconstructs text from the unified semantic space
- Pre-trained checkpoints and example data

### `data_process/`
Contains dataset-specific data processing scripts:
- `Broderick2018/`: Data processing for Broderick 2018 dataset
- `LPPC-fMRI/`: Data processing for LPPC-fMRI dataset
- `SMN4Lang_fMRI/`: Data processing for SMN4Lang fMRI recordings
- `SMN4Lang_MEG/`: Data processing for SMN4Lang MEG recordings
- `SMN4Lang_multi/`: Multi-modal data processing scripts

### `models/`
Contains model implementations:
- EEG-based models (EEG_MLP_multisub_EEGPT_v2.py, EEG_feature_extra_EEGPT.py)
- fMRI-based models (fMRIcortical_MLP_new_multisubject_v2.py, fMRI_cortical_pretrain_MAE_v2.py)
- MEG-based model (MEG_MLP_multisub.py)
- Utilities for masking, metrics, and time features

### `utils/`
Utility functions for:
- Data alignment and preprocessing
- Embedding extraction and feature processing
- Loss functions and evaluation metrics
- Text processing and word tokenization
- Brain data handling (EEG, fMRI, MEG)

## Quick Start

### Testing with Demo Data

Navigate to the demo directory and run the test:

```bash
cd demo/CN_test  # or EN_test for English
python get_Uniemb.py
python gen_text_main.py
```

This will:
1. Load pre-trained brain-to-semantic mappings
2. Process sample brain recordings
3. Map them to the unified semantic space using `get_Uniemb.py`
4. Reconstruct and output decoded text

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon multiple neuroimaging datasets and pre-trained multilingual language models. We thank all participants and institutions involved in data collection.

## Support

For questions or issues, please open an issue on the repository.
