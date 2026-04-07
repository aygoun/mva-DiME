# Download the ESC-50 dataset
git clone https://github.com/karolpiczak/ESC-50.git
mv ESC-50 dataset/ESC-50-master

# Download the DDPM model
wget https://huggingface.co/guillaumejs2403/DiME/resolve/main/ddpm-celeba.pt
mkdir -p models
mv ddpm-celeba.pt models/ddpm-celeba.pt

# Download the fine-tuned classifier model
mkdir -p audio/models
hf download armandblin/DiME_audio --repo-type model -o audio/models --include="*"
