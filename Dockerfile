FROM python:3.11
WORKDIR mazeGPT
COPY . .
RUN pip install -r requirements.txt
RUN useradd -m -u 1000 user
USER user
CMD ["bash"]