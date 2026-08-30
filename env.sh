# Is project ke liye shell setup. Har script chalane se pehle:  source env.sh
#
# GPU 10 use karni hai (user instruction). Lekin torch ka device index
# nvidia-smi ke index se match NAHI karta (beech me ek T1000 Exclusive_Process
# mode me hai, jisko chhoote hi "CUDA-capable device is busy" error aata hai).
# Isliye PCI_BUS_ID order + CUDA_VISIBLE_DEVICES se GPU 10 ko isolate karte
# hain, aur code ke andar wo hamesha cuda:0 hoti hai.

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=10
export PY=/speech/abhishek/miniconda3/envs/grain/bin/python

# verify:  $PY -c "import torch;print(torch.cuda.get_device_name(0))"
