# chatnmi_v2

## Dependencies Installation
The torch library is required for CUDA. Make sure that the correct version is installed.

    pip install torch torchvision  --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
    pip install .

    pip uninstall -y autoawq
    git clone https://github.com/casper-hansen/AutoAWQ
    cd AutoAWQ
    pip install .

To run Mixtral model, the special version of llama-cpp is required.
https://medium.com/@piyushbatra1999/installing-llama-cpp-python-with-nvidia-gpu-acceleration-on-windows-a-short-guide-0dfac475002d
https://pypi.org/project/llama-cpp-python/
