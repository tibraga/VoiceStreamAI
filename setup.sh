#!/bin/bash

# Instala o pacote ctranslate2
python3 -m pip install ctranslate2==4.4.0

# Calcula o valor de LD_LIBRARY_PATH e exporta a variável
export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> ~/.bashrc
