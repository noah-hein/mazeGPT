FROM python:3.11
WORKDIR mazeGPT
COPY . .
RUN pip3 install torch torchvision torchaudio
RUN pip install -r requirements.txt
RUN useradd -m -u 1000 user
USER user
CMD ["bash"]