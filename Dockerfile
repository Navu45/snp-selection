FROM quay.io/jupyter/all-spark-notebook

USER root

RUN apt-get update

RUN apt-get install -y \
    openjdk-11-jre-headless \
    g++ \
    python3 python3-pip \
    libopenblas-base liblapack3

RUN python3 -m pip install hail

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]