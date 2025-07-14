FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pip.conf /etc/pip.conf

WORKDIR /app

COPY app/ .
COPY requirements.txt .
COPY yolo11n.pt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --timeout=1000 && \
    rm -fr requirements.txt /etc/pip.conf

EXPOSE 8080

CMD [ "python3", "main.py" ]